#!/usr/bin/env python3
"""Build an opt-in dynamic INT8 FlowLM G5 source graph and check host parity.

Run in the pinned Linux conversion environment. The default position-major
variant keeps float32 K/V I/O for ``npuPositionMajorCache=true``. Compile the
result separately in the LiteRT 2.2.0 AOT environment with, for example::

    python scripts/aot_tensor_g5.py pt_flowlm_fused_dyn8_all_contiguous \
        --truncation no_truncation

The existing group-major CPU ``dyn8_all`` graph is separate; this script never
replaces a production graph.
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
CANDIDATE_STEM = "pt_flowlm_fused_dyn8_all_contiguous"


def signature(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    return [inputs, outputs]


def check_interface(source: Path, quantized: Path) -> None:
    expected, actual = signature(source), signature(quantized)
    for side, original, candidate in zip(("input", "output"), expected, actual):
        if len(original) != len(candidate):
            raise AssertionError(f"{side} count changed: {len(original)} -> {len(candidate)}")
        for i, (a, b) in enumerate(zip(original, candidate)):
            a_shape = tuple(int(x) for x in a["shape"])
            b_shape = tuple(int(x) for x in b["shape"])
            if a_shape != b_shape or a["dtype"] != b["dtype"]:
                raise AssertionError(f"{side} {i} changed: {a_shape}/{a['dtype']} -> {b_shape}/{b['dtype']}")
            if b["dtype"] != np.float32:
                raise AssertionError(f"{side} {i} must remain float32 for the Android cache path")
            print(f"{side}[{i}] {b_shape} {b['dtype'].__name__}")
    kv_shape = (1, bp.PMAX, bp.N_LAYERS * bp.N_HEADS, bp.HD)
    for i in (4, 5):
        if tuple(actual[0][i]["shape"]) != kv_shape:
            raise AssertionError(f"cache input {i} has wrong position-major shape")


def check_short_rollout(path: Path, steps: int) -> None:
    """Compare a short free run with eager fp32; this is a latent quality proxy."""
    model = bp.load_eager()
    flow_lm = model.flow_lm
    reference = bp.FusedStep(flow_lm).eval()
    keys, values, pos = bp.load_voice_state("alba")
    pk_ref, pv_ref = bp.pack_voice(keys, values, pos)
    pk_q = np.ascontiguousarray(pk_ref.numpy().transpose(0, 2, 1, 3))
    pv_q = np.ascontiguousarray(pv_ref.numpy().transpose(0, 2, 1, 3))
    in_w = flow_lm.input_linear.weight.detach()
    x_ref = (flow_lm.bos_emb.detach() @ in_w.T).view(1, 1, -1)
    x_q = x_ref.numpy().copy()
    runner = bp.CM(str(path))
    torch.manual_seed(3)
    latent_corr, latent_delta, kv_delta = [], [], []
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
            if step == 0:
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
            kv_delta.append(bp.maxd(got[1 + bp.LDIM:], ref[1 + bp.LDIM:]))
            print(
                f"step {step} position={pos} eos_delta={abs(float(got[0] - ref[0])):.3e} "
                f"latent_corr={latent_corr[-1]:.8f} latent_max_delta={latent_delta[-1]:.3e} "
                f"kv_max_delta={kv_delta[-1]:.3e}"
            )
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
        f"short rollout: {steps} steps; min_latent_corr={min(latent_corr):.8f} "
        f"max_latent_delta={max(latent_delta):.3e} max_kv_delta={max(kv_delta):.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("PT_OUT", bp.OUT)))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--steps", type=int, default=4, help="host free-run parity steps")
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 32:
        parser.error("--steps must be in 1..32")
    source = args.out / f"{SOURCE_STEM}.tflite"
    candidate = args.out / f"{CANDIDATE_STEM}.tflite"
    if not source.is_file():
        parser.error(f"missing source graph: {source}")
    if not args.check_only:
        bp.to_quant(str(source), str(candidate), bp.QUANT_VARIANTS["dyn8_all"])
    if not candidate.is_file():
        parser.error(f"missing candidate graph: {candidate}")
    check_interface(source, candidate)
    check_short_rollout(candidate, args.steps)


if __name__ == "__main__":
    main()
