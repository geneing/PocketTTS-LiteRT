#!/usr/bin/env python3
"""Export a fused FlowLM step that returns its updated K/V cache banks.

This is an opt-in Tensor G5 prototype. Run it in the pinned Linux conversion
environment from AGENTS.md, then compile the fp16 graph with:

  PT_OUT=<model-dir> python scripts/aot_tensor_g5.py \
      pt_flowlm_fused_fp16_resident --truncation no_truncation

The AOT file is named `pt_flowlm_fused_fp16_resident_no_truncation_g5.tflite`.
On Android, pass the corresponding logical base name ending in
`_no_truncation.tflite` with `Accel.NPU` and `npuResidentCache=true`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_pockettts as bp  # noqa: E402


FP32_NAME = "pt_flowlm_fused_fp32_resident.tflite"
FP16_NAME = "pt_flowlm_fused_fp16_resident.tflite"
_MODEL = None


class ResidentFusedStep(nn.Module):
    """One FlowLM frame plus a full, updated pair of static K/V cache tensors.

    Cache tensors stay separate from the small EOS/latent output so the Android
    runner can chain the cache buffers without reading 25 MiB back to the host.
    `write_mask` is a [1,1,PMAX,1] one-hot mask for the current cache position.
    """

    def __init__(self, flow_lm, cache_update="where"):
        super().__init__()
        self.step = bp.FlowLMStep(flow_lm)
        self.head = bp.FlowHead(flow_lm)
        self.cache_update = cache_update

    def forward(self, x, cos, sin, mask, pk, pv, noise, write_mask):
        cond, eos, new_k, new_v = self.step(x, cos, sin, mask, pk, pv)
        latent = self.head(cond, noise)
        if self.cache_update == "arithmetic":
            keep = 1.0 - write_mask
            next_k = pk * keep + new_k * write_mask
            next_v = pv * keep + new_v * write_mask
        else:
            write = write_mask > 0.5
            next_k = torch.where(write, new_k, pk)
            next_v = torch.where(write, new_v, pv)
        control = torch.cat([eos, latent], dim=-1)
        return control, next_k, next_v


def _inputs(model):
    flm = model.flow_lm
    keys, values, pos = bp.load_voice_state("alba")
    pk, pv = bp.pack_voice(keys, values, pos)
    emb = (flm.bos_emb.detach() @ flm.input_linear.weight.detach().T).view(1, 1, -1)
    cos, sin = bp.rope_cos_sin_deint(pos)
    cos = torch.from_numpy(cos).view(1, 1, 1, bp.HD)
    sin = torch.from_numpy(sin).view(1, 1, 1, bp.HD)
    mask = torch.from_numpy(bp.make_mask(pos))
    torch.manual_seed(3)
    noise = torch.randn(1, bp.LDIM) * bp.math.sqrt(0.3)
    write_mask = torch.zeros((1, 1, bp.PMAX, 1), dtype=torch.float32)
    write_mask[:, :, pos, :] = 1.0
    args = (emb, cos, sin, mask, pk, pv, noise, write_mask)
    return pos, args


def _verify_fp32(module, args, out: Path) -> None:
    """Run the exported CPU graph once and compare all three outputs to eager."""
    from ai_edge_litert.compiled_model import CompiledModel

    # Recover the selected index instead of inferring it from the number of set
    # entries; this also asserts the write mask is truly one-hot.
    write_positions = torch.nonzero(args[-1].reshape(-1), as_tuple=False).reshape(-1)
    if write_positions.numel() != 1:
        raise AssertionError(f"expected one write position, got {write_positions.numel()}")
    pos = int(write_positions.item())

    with torch.no_grad():
        eager = module(*args)
        reference = bp.FusedStep(_MODEL.flow_lm).eval()(*args[:7])
        # The wrapper's output includes only the values needed by the host; the
        # fp32 source graph must reproduce both them and the full cache update.
        expected_k = torch.where(args[-1] > 0.5, reference[:, 1 + bp.LDIM:
            1 + bp.LDIM + bp.G_KV].reshape(1, bp.N_LAYERS * bp.N_HEADS, 1, bp.HD), args[4])
        expected_v = torch.where(args[-1] > 0.5, reference[:, 1 + bp.LDIM + bp.G_KV:
            ].reshape(1, bp.N_LAYERS * bp.N_HEADS, 1, bp.HD), args[5])
        expected = (reference[:, :1 + bp.LDIM], expected_k, expected_v)
        for name, got, want in zip(("control", "next_k", "next_v"), eager, expected):
            delta = bp.maxd(got.numpy(), want.numpy())
            if delta != 0.0:
                raise AssertionError(f"eager cache wrapper {name} mismatch: max|d|={delta}")

    flat_args = [np.ascontiguousarray(x.detach().numpy(), dtype=np.float32) for x in args]
    sizes = [1 + bp.LDIM, args[4].numel(), args[5].numel()]
    actual = bp.run_signature(str(out), 0, flat_args, sizes)
    for name, got, want in zip(("control", "next_k", "next_v"), actual, expected):
        want_np = want.detach().numpy().reshape(-1)
        delta = bp.maxd(got, want_np)
        corr = bp.corr(got, want_np) if got.size > 1 else 1.0
        print(f"CPU fp32 parity {name}: corr={corr:.8f} max|d|={delta:.3e}")
        if delta > 2e-5:
            raise AssertionError(f"exported fp32 {name} mismatch: max|d|={delta}")
    print(f"verified one-hot K/V update at position {pos}")


def export(out: Path, cache_update: str = "where") -> None:
    global _MODEL
    out.mkdir(parents=True, exist_ok=True)
    print("Loading pinned Pocket TTS eager model...")
    _MODEL = bp.load_eager()
    module = ResidentFusedStep(_MODEL.flow_lm, cache_update).eval()
    pos, args = _inputs(_MODEL)
    suffix = "" if cache_update == "where" else f"_{cache_update}"
    fp32 = out / FP32_NAME.replace(".tflite", f"{suffix}.tflite")
    fp16 = out / FP16_NAME.replace(".tflite", f"{suffix}.tflite")

    print("Checking eager output and cache-update shapes...")
    with torch.no_grad():
        control, next_k, next_v = module(*args)
    assert tuple(control.shape) == (1, 1 + bp.LDIM)
    assert tuple(next_k.shape) == tuple(args[4].shape)
    assert tuple(next_v.shape) == tuple(args[5].shape)
    print(
        f"eager position={pos}; control={tuple(control.shape)}; "
        f"K/V={tuple(next_k.shape)}; host-readable output={control.numel() * 4} bytes"
    )

    bp.convert(module, args, str(fp32))
    _verify_fp32(module, args, fp32)
    bp.to_fp16(str(fp32), str(fp16))
    print(f"AOT input: {fp16} ({fp16.stat().st_size} bytes)")
    print(
        "Next: compile the fp16 model for Tensor G5 with "
        "aot_tensor_g5.py pt_flowlm_fused_fp16_resident --truncation no_truncation."
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("PT_OUT", HERE / "out")),
        help="directory for the source and fp16 graphs (default: PT_OUT or scripts/out)",
    )
    parser.add_argument("--cache-update", choices=("where", "arithmetic"), default="where")
    args = parser.parse_args()
    export(args.out, args.cache_update)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
