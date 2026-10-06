#!/usr/bin/env python3
"""Build opt-in quantized FlowLM G5 source graphs and check host parity.

Run in the pinned Linux conversion environment. The default position-major
variant keeps float32 K/V I/O for ``npuPositionMajorCache=true``. Compile the
result separately in the LiteRT 2.2.0 AOT environment with, for example::

    python scripts/aot_tensor_g5.py pt_flowlm_fused_dyn8_all_contiguous \
        --truncation no_truncation

The existing group-major CPU ``dyn8_all`` graph is separate; this script never
replaces a production graph.

``--recipe static16`` tries calibrated channelwise W8/A16 on the same source.
It transposes the representative K/V caches to position-major order and reports
whether the external float32 cache protocol survives quantization.

``--recipe static8`` uses AEQ 0.8.0's calibrated ``static_wi8_ai8`` recipe,
explicitly leaving every external input and output float32. Calibration walks
an eager free run from pinned preset voice states, sampling positions and K/V
cache magnitudes. The output is opt-in and is never installed by this script.

``--recipe static16_floatio`` applies the corresponding W8/A16 recipe with
the same float32 host interface. It is an isolated NPU quality experiment.
``--recipe static16_fc_floatio`` limits W8/A16 to fully connected operations.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

import build_pockettts as bp


SOURCE_STEM = "pt_flowlm_fused_fp32_contiguous"
CANDIDATE_STEMS = {
    "dynamic8": "pt_flowlm_fused_dyn8_all_contiguous",
    "static16": "pt_flowlm_fused_st16_all_contiguous",
    "static8": "pt_flowlm_fused_st8_floatio_contiguous",
    "static16_floatio": "pt_flowlm_fused_st16_floatio_contiguous",
    "static16_fc_floatio": "pt_flowlm_fused_st16_fc_floatio_contiguous",
}


def signature(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    return [inputs, outputs]


def check_interface(source: Path, quantized: Path, require_float_io: bool) -> None:
    expected, actual = signature(source), signature(quantized)
    float_io = True
    for side, original, candidate in zip(("input", "output"), expected, actual):
        if len(original) != len(candidate):
            raise AssertionError(f"{side} count changed: {len(original)} -> {len(candidate)}")
        for i, (a, b) in enumerate(zip(original, candidate)):
            a_shape = tuple(int(x) for x in a["shape"])
            b_shape = tuple(int(x) for x in b["shape"])
            if a_shape != b_shape:
                raise AssertionError(f"{side} {i} changed: {a_shape}/{a['dtype']} -> {b_shape}/{b['dtype']}")
            if a["dtype"] != np.float32:
                raise AssertionError(f"source {side} {i} is not float32: {a['dtype']}")
            if b["dtype"] != np.float32:
                float_io = False
            print(f"{side}[{i}] {b_shape} {b['dtype'].__name__}")
    kv_shape = (1, bp.PMAX, bp.N_LAYERS * bp.N_HEADS, bp.HD)
    for i in (4, 5):
        if tuple(actual[0][i]["shape"]) != kv_shape:
            raise AssertionError(f"cache input {i} has wrong position-major shape")
    if tuple(actual[1][0]["shape"]) != (1, 1 + bp.LDIM + 2 * bp.G_KV):
        raise AssertionError("packed EOS/latent/K/V output shape changed")
    if require_float_io and not float_io:
        raise AssertionError("graph changed float32 I/O required by the Android cache path")
    if not float_io:
        print("WARNING: quantized external I/O is incompatible with the current Android float cache path")


def inspect_internal_quantization(path: Path, activation_bits: int) -> None:
    from collections import Counter
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    tensors = Counter(str(detail["dtype"]) for detail in interpreter.get_tensor_details())
    ops = Counter(detail["op_name"] for detail in interpreter._get_ops_details())
    print(f"internal tensor dtypes: {dict(tensors)}")
    print(f"quantization ops: QUANTIZE={ops['QUANTIZE']} DEQUANTIZE={ops['DEQUANTIZE']}")
    activation_type = f"int{activation_bits}"
    if not any(activation_type in dtype for dtype in tensors) or ops["QUANTIZE"] == 0:
        raise AssertionError(f"static W8/A{activation_bits} recipe did not quantize activations")


def quantize_static_floatio(
    source: Path, candidate: Path, samples: list[dict], activation_bits: int,
    fully_connected_only: bool = False,
) -> None:
    from ai_edge_quantizer import algorithm_manager, calibrator, qtyping, quantizer

    qt = quantizer.Quantizer(float_model=str(source))
    if fully_connected_only:
        qt.load_quantization_recipe(bp.quant_recipe({
            "kind": "static", "ops": ["FULLY_CONNECTED"],
            "weight_bits": 8, "act_bits": activation_bits, "regex": ".*",
        }))
    else:
        qt.load_quantization_recipe(f"static_wi8_ai{activation_bits}")
    for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
        qt.update_quantization_recipe(
            regex=".*", operation_name=op,
            algorithm_key=algorithm_manager.AlgorithmName.NO_QUANTIZE,
        )
    calibration_result = qt.calibrate(
        {"serving_default": samples},
        mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
    )
    qt.quantize(calibration_result=calibration_result).export_model(str(candidate))
    print(f"exported {candidate} ({candidate.stat().st_size / 1e6:.1f} MB)")


class InterpreterRunner:
    """Host parity fallback for int16 I/O unsupported by Python TensorBuffer.write."""

    def __init__(self, path: Path):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()

    def __call__(self, *arrays):
        for array, detail in zip(arrays, self.inputs):
            value = bp.quant_to(array, detail["dtype"], detail["quantization"])
            self.interpreter.set_tensor(detail["index"], np.ascontiguousarray(value))
        self.interpreter.invoke()
        return [bp.quant_from(
            self.interpreter.get_tensor(detail["index"]),
            detail["dtype"], detail["quantization"],
        ) for detail in self.outputs]


class GroupStepRunner:
    """Run the shipped group-major graph's step signature (prefill is index 0)."""

    def __init__(self, path: Path):
        from ai_edge_litert.compiled_model import CompiledModel

        self.model = CompiledModel.from_file(str(path))
        self.index = 1
        self.inputs = self.model.create_input_buffers(self.index)
        self.outputs = self.model.create_output_buffers(self.index)

    def __call__(self, *arrays):
        for buffer, array in zip(self.inputs, arrays):
            buffer.write(np.ascontiguousarray(array, dtype=np.float32).ravel())
        self.model.run_by_index(self.index, self.inputs, self.outputs)
        return [np.array(self.outputs[0].read(1 + bp.LDIM + 2 * bp.G_KV, np.float32))]


def check_short_rollout(
    path: Path, steps: int, compare_group_dyn8: bool,
    baseline_graph: Path | None = None, voice: str = "alba",
) -> None:
    """Compare a free run with eager fp32, including recurrent cache drift."""
    model = bp.load_eager()
    flow_lm = model.flow_lm
    reference = bp.FusedStep(flow_lm).eval()
    keys, values, pos = bp.load_voice_state(voice)
    if pos + steps > bp.PMAX:
        raise ValueError(f"voice position {pos} + {steps} steps exceeds cache {bp.PMAX}")
    pk_ref, pv_ref = bp.pack_voice(keys, values, pos)
    pk_q = np.ascontiguousarray(pk_ref.numpy().transpose(0, 2, 1, 3))
    pv_q = np.ascontiguousarray(pv_ref.numpy().transpose(0, 2, 1, 3))
    in_w = flow_lm.input_linear.weight.detach()
    x_ref = (flow_lm.bos_emb.detach() @ in_w.T).view(1, 1, -1)
    x_q = x_ref.numpy().copy()
    float_io = all(d["dtype"] == np.float32 for side in signature(path) for d in side)
    runner = bp.CM(str(path)) if float_io else InterpreterRunner(path)
    baseline_runner = GroupStepRunner(baseline_graph) if baseline_graph else None
    if baseline_runner:
        pk_base, pv_base = pk_ref.numpy().copy(), pv_ref.numpy().copy()
        x_base = x_ref.numpy().copy()
        baseline_corr, baseline_lat_delta, baseline_eos_delta, baseline_kv_delta = [], [], [], []
    torch.manual_seed(3)
    latent_corr, latent_delta, eos_delta, kv_delta = [], [], [], []
    with torch.no_grad():
        for step in range(steps):
            cos, sin = bp.rope_cos_sin_deint(pos)
            mask = bp.make_mask(pos)
            noise = torch.randn(1, bp.LDIM) * math.sqrt(0.3)
            ref = reference(
                x_ref, torch.from_numpy(cos).view(1, 1, 1, bp.HD),
                torch.from_numpy(sin).view(1, 1, 1, bp.HD),
                torch.from_numpy(mask), pk_ref, pv_ref, noise,
            ).numpy().reshape(-1)
            got = runner(
                x_q, cos.reshape(1, 1, 1, bp.HD), sin.reshape(1, 1, 1, bp.HD),
                mask, pk_q, pv_q, noise.numpy(),
            )[0].reshape(-1)
            if step == 0 and compare_group_dyn8:
                group_graph = path.parent / "pt_flowlm_fused_dyn8_all.tflite"
                if group_graph.is_file():
                    group = bp.run_signature(
                        str(group_graph), 1,
                        (x_ref.numpy(), cos.reshape(1, 1, 1, bp.HD),
                         sin.reshape(1, 1, 1, bp.HD), mask,
                         pk_ref.numpy(), pv_ref.numpy(), noise.numpy()),
                        [ref.size],
                    )[0].reshape(-1)
                    full_delta = bp.maxd(got, group)
                    print(
                        f"position versus shipped group dyn8, first step: "
                        f"full_max_delta={full_delta:.3e} "
                        f"latent_corr={bp.corr(got[1:1 + bp.LDIM], group[1:1 + bp.LDIM]):.8f} "
                        f"latent_max_delta={bp.maxd(got[1:1 + bp.LDIM], group[1:1 + bp.LDIM]):.3e} "
                        f"kv_max_delta={bp.maxd(got[1 + bp.LDIM:], group[1 + bp.LDIM:]):.3e}"
                    )
                    if full_delta > 1e-3:
                        raise AssertionError("position-major dyn8 differs from shipped group-major dyn8")
            if not np.isfinite(got).all():
                raise AssertionError(f"nonfinite output at step {step}")
            latent_ref = ref[1:1 + bp.LDIM]
            latent_got = got[1:1 + bp.LDIM]
            latent_corr.append(bp.corr(latent_got, latent_ref))
            latent_delta.append(bp.maxd(latent_got, latent_ref))
            eos_delta.append(abs(float(got[0] - ref[0])))
            kv_delta.append(bp.maxd(got[1 + bp.LDIM:], ref[1 + bp.LDIM:]))
            print(
                f"step {step} position={pos} eos_delta={eos_delta[-1]:.3e} "
                f"latent_corr={latent_corr[-1]:.8f} latent_max_delta={latent_delta[-1]:.3e} "
                f"kv_max_delta={kv_delta[-1]:.3e}"
            )
            if baseline_runner:
                base = baseline_runner(
                    x_base, cos.reshape(1, 1, 1, bp.HD),
                    sin.reshape(1, 1, 1, bp.HD), mask,
                    pk_base, pv_base, noise.numpy(),
                )[0].reshape(-1)
                if not np.isfinite(base).all():
                    raise AssertionError(f"nonfinite CPU dyn8 output at step {step}")
                base_lat = base[1:1 + bp.LDIM]
                baseline_corr.append(bp.corr(base_lat, latent_ref))
                baseline_lat_delta.append(bp.maxd(base_lat, latent_ref))
                baseline_eos_delta.append(abs(float(base[0] - ref[0])))
                baseline_kv_delta.append(bp.maxd(base[1 + bp.LDIM:], ref[1 + bp.LDIM:]))
                pk_base[0, :, pos] = base[1 + bp.LDIM:1 + bp.LDIM + bp.G_KV].reshape(-1, bp.HD)
                pv_base[0, :, pos] = base[1 + bp.LDIM + bp.G_KV:].reshape(-1, bp.HD)
                x_base = (base_lat @ in_w.numpy().T).reshape(1, 1, -1)
            new_k_ref = ref[1 + bp.LDIM:1 + bp.LDIM + bp.G_KV].reshape(-1, bp.HD)
            new_v_ref = ref[1 + bp.LDIM + bp.G_KV:].reshape(-1, bp.HD)
            new_k_q = got[1 + bp.LDIM:1 + bp.LDIM + bp.G_KV].reshape(-1, bp.HD)
            new_v_q = got[1 + bp.LDIM + bp.G_KV:].reshape(-1, bp.HD)
            pk_ref[0, :, pos] = torch.from_numpy(new_k_ref)
            pv_ref[0, :, pos] = torch.from_numpy(new_v_ref)
            pk_q[0, pos] = new_k_q
            pv_q[0, pos] = new_v_q
            x_ref = (torch.from_numpy(latent_ref) @ in_w.T).view(1, 1, -1)
            x_q = (latent_got @ in_w.numpy().T).reshape(1, 1, -1)
            pos += 1
    print(
        f"rollout: voice={voice} {steps} steps; min_latent_corr={min(latent_corr):.8f} "
        f"max_latent_delta={max(latent_delta):.3e} max_eos_delta={max(eos_delta):.3e} "
        f"max_kv_delta={max(kv_delta):.3e}"
    )
    if baseline_runner:
        print(
            f"CPU dyn8 baseline: min_latent_corr={min(baseline_corr):.8f} "
            f"max_latent_delta={max(baseline_lat_delta):.3e} "
            f"max_eos_delta={max(baseline_eos_delta):.3e} "
            f"max_kv_delta={max(baseline_kv_delta):.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=CANDIDATE_STEMS, default="dynamic8")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("PT_OUT", bp.OUT)))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--steps", type=int, help="host free-run parity steps (float-I/O static default 32; others 4)")
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--calibration-run", type=int, default=128)
    parser.add_argument("--calibration-voices", default="alba",
                        help="comma-separated pinned preset voices; sample count and run length apply per voice")
    parser.add_argument("--calibration-seed", type=int, default=11)
    parser.add_argument("--voice", default="alba", help="voice for host rollout")
    parser.add_argument("--compare-dyn8", type=Path,
                        help="optional group-major CPU dyn8 graph for a paired free-run baseline")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 32 if args.recipe in ("static8", "static16_floatio", "static16_fc_floatio") else 4
    if args.steps < 1 or args.steps > bp.PMAX:
        parser.error(f"--steps must be in 1..{bp.PMAX}")
    source = args.out / f"{SOURCE_STEM}.tflite"
    candidate = args.out / f"{CANDIDATE_STEMS[args.recipe]}.tflite"
    if not source.is_file():
        parser.error(f"missing source graph: {source}")
    if not args.check_only:
        if args.recipe == "dynamic8":
            bp.to_quant(str(source), str(candidate), bp.QUANT_VARIANTS["dyn8_all"])
        else:
            if args.calibration_samples < 1 or args.calibration_run < args.calibration_samples:
                parser.error("static recipe needs 1 <= calibration-samples <= calibration-run")
            voices = [voice.strip() for voice in args.calibration_voices.split(",") if voice.strip()]
            if not voices:
                parser.error("--calibration-voices must name at least one voice")
            model = bp.load_eager()
            input_details = signature(source)[0]
            samples = []
            for voice_index, voice in enumerate(voices):
                voice_samples = bp.quant_calibration(
                    model, voice=voice, n=args.calibration_samples,
                    run=args.calibration_run, seed=args.calibration_seed + voice_index,
                )[:args.calibration_samples]
                for sample in voice_samples:
                    for key in ("args_4", "args_5"):
                        sample[key] = np.ascontiguousarray(sample[key].transpose(0, 2, 1, 3))
                    for index, detail in enumerate(input_details):
                        value = sample[f"args_{index}"]
                        if value.dtype != np.float32 or tuple(value.shape) != tuple(detail["shape"]):
                            raise AssertionError(
                                f"calibration {voice} args_{index}: {value.shape}/{value.dtype} "
                                f"expected {detail['shape']}/{detail['dtype']}"
                            )
                samples.extend(voice_samples)
            print(f"calibration settings: voices={voices} seed={args.calibration_seed} "
                  f"samples={len(samples)} run_per_voice={args.calibration_run}")
            candidate.unlink(missing_ok=True)
            if args.recipe in ("static8", "static16_floatio", "static16_fc_floatio"):
                activation_bits = 8 if args.recipe == "static8" else 16
                quantize_static_floatio(
                    source, candidate, samples, activation_bits,
                    fully_connected_only=args.recipe == "static16_fc_floatio",
                )
            else:
                bp.to_quant(
                    str(source), str(candidate), bp.QUANT_VARIANTS["st16_all"],
                    calibration={"serving_default": samples},
                )
            del samples, model
    if not candidate.is_file():
        parser.error(f"missing candidate graph: {candidate}")
    check_interface(source, candidate, require_float_io=args.recipe != "static16")
    if args.recipe in ("static8", "static16_floatio", "static16_fc_floatio"):
        inspect_internal_quantization(candidate, 8 if args.recipe == "static8" else 16)
    if args.compare_dyn8 and not args.compare_dyn8.is_file():
        parser.error(f"missing CPU dyn8 baseline graph: {args.compare_dyn8}")
    check_short_rollout(candidate, args.steps,
                        compare_group_dyn8=args.recipe == "dynamic8",
                        baseline_graph=args.compare_dyn8, voice=args.voice)


if __name__ == "__main__":
    main()
