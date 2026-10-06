#!/usr/bin/env python3
"""Export the fused FlowLM step with position-major K/V inputs for Tensor G5.

The graph accepts K/V [1, PMAX, G, HD] and produces the same packed
eos/latent/new-K/new-V output as the group-major one-step graph. Host row
updates then touch one contiguous G*HD slab in each persistent input buffer.

In the pinned Linux environments:
  python scripts/export_flowlm_npu_contiguous.py
  python scripts/aot_tensor_g5.py pt_flowlm_fused_fp16_contiguous \
      --truncation no_truncation
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


class PositionMajorFusedStep(nn.Module):
    def __init__(self, flow_lm):
        super().__init__()
        self.fused = bp.FusedStep(flow_lm)

    def forward(self, emb, cos, sin, mask, pk_pos, pv_pos, noise):
        # The existing attention math still consumes [1,G,PMAX,HD]. The
        # transpose is in the compiled graph, not performed on the host.
        return self.fused(
            emb, cos, sin, mask,
            pk_pos.transpose(1, 2), pv_pos.transpose(1, 2), noise,
        )


def export(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model = bp.load_eager()
    flow_lm = model.flow_lm
    keys, values, pos = bp.load_voice_state("alba")
    pk, pv = bp.pack_voice(keys, values, pos)
    pk_pos = pk.transpose(1, 2).contiguous()
    pv_pos = pv.transpose(1, 2).contiguous()
    emb = (flow_lm.bos_emb.detach() @ flow_lm.input_linear.weight.detach().T).view(1, 1, -1)
    cos, sin = bp.rope_cos_sin_deint(pos)
    cos = torch.from_numpy(cos).view(1, 1, 1, bp.HD)
    sin = torch.from_numpy(sin).view(1, 1, 1, bp.HD)
    mask = torch.from_numpy(bp.make_mask(pos))
    torch.manual_seed(3)
    noise = torch.randn(1, bp.LDIM) * bp.math.sqrt(0.3)
    args = (emb, cos, sin, mask, pk_pos, pv_pos, noise)

    module = PositionMajorFusedStep(flow_lm).eval()
    reference = bp.FusedStep(flow_lm).eval()
    with torch.no_grad():
        want = reference(emb, cos, sin, mask, pk, pv, noise).numpy()
        got = module(*args).numpy()
    assert tuple(pk_pos.shape) == (1, bp.PMAX, bp.N_LAYERS * bp.N_HEADS, bp.HD)
    assert np.array_equal(pk_pos.numpy().transpose(0, 2, 1, 3), pk.numpy())
    assert np.array_equal(pv_pos.numpy().transpose(0, 2, 1, 3), pv.numpy())
    assert np.array_equal(got, want), f"eager max|d|={bp.maxd(got, want)}"
    print(f"eager position-major parity exact at voice position {pos}")

    fp32 = out / "pt_flowlm_fused_fp32_contiguous.tflite"
    fp16 = out / "pt_flowlm_fused_fp16_contiguous.tflite"
    bp.convert(module, args, str(fp32))
    flat_args = [np.ascontiguousarray(x.detach().numpy(), dtype=np.float32) for x in args]
    actual = bp.run_signature(str(fp32), 0, flat_args, [want.size])[0]
    delta = bp.maxd(actual, want.reshape(-1))
    corr = bp.corr(actual, want.reshape(-1))
    print(f"CPU fp32 graph parity: corr={corr:.8f} max|d|={delta:.3e}")
    if delta > 2e-5:
        raise AssertionError(f"position-major CPU graph mismatch: {delta}")

    # Patch the emitted row using the same contiguous [position,G,HD] slab as
    # the native bridge, then verify the next token attends to the same cache.
    row_floats = bp.N_LAYERS * bp.N_HEADS * bp.HD
    new_k = actual[1 + bp.LDIM:1 + bp.LDIM + row_floats].reshape(-1, bp.HD)
    new_v = actual[1 + bp.LDIM + row_floats:].reshape(-1, bp.HD)
    pk_next, pv_next = pk.clone(), pv.clone()
    pk_pos_next, pv_pos_next = pk_pos.clone(), pv_pos.clone()
    pk_next[0, :, pos] = torch.from_numpy(new_k)
    pv_next[0, :, pos] = torch.from_numpy(new_v)
    pk_pos_next[0, pos] = torch.from_numpy(new_k)
    pv_pos_next[0, pos] = torch.from_numpy(new_v)
    assert np.array_equal(pk_pos_next.numpy().transpose(0, 2, 1, 3), pk_next.numpy())
    assert np.array_equal(pv_pos_next.numpy().transpose(0, 2, 1, 3), pv_next.numpy())
    cos2, sin2 = bp.rope_cos_sin_deint(pos + 1)
    cos2 = torch.from_numpy(cos2).view(1, 1, 1, bp.HD)
    sin2 = torch.from_numpy(sin2).view(1, 1, 1, bp.HD)
    mask2 = torch.from_numpy(bp.make_mask(pos + 1))
    with torch.no_grad():
        want2 = reference(emb, cos2, sin2, mask2, pk_next, pv_next, noise).numpy()
    args2 = (emb, cos2, sin2, mask2, pk_pos_next, pv_pos_next, noise)
    flat_args2 = [np.ascontiguousarray(x.detach().numpy(), dtype=np.float32) for x in args2]
    actual2 = bp.run_signature(str(fp32), 0, flat_args2, [want2.size])[0]
    delta2 = bp.maxd(actual2, want2.reshape(-1))
    print(f"CPU fp32 next-step cache parity: corr={bp.corr(actual2, want2.reshape(-1)):.8f} max|d|={delta2:.3e}")
    if delta2 > 2e-5:
        raise AssertionError(f"position-major next-step cache mismatch: {delta2}")
    bp.to_fp16(str(fp32), str(fp16))
    print(f"AOT input: {fp16} ({fp16.stat().st_size} bytes)")


if __name__ == "__main__":
    export(Path(os.environ.get("PT_OUT", HERE / "out")))
