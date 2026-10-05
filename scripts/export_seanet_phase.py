#!/usr/bin/env python3
"""Export the phase-packed SEANet decoder experiment for Tensor G5.

Run with the conversion environment and the pinned Pocket TTS reference clone:
``PT_STREAM_W=512 python scripts/export_seanet_phase.py``.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import build_pockettts as pt


def main() -> None:
    model = pt.load_eager()
    windows = [int(w) for w in os.environ.get("PT_STREAM_W", "512").split(",")]
    for width in windows:
        print(f"\n=== phase-packed SEANet, W={width} ===")
        torch.manual_seed(41)
        x = torch.randn(1, pt.MIMI_D, width) * 0.5
        zero = pt.MimiDecOnly(model, L=width).eval()
        phase = pt.MimiDecOnly(model, L=width, convtr_impl="phase").eval()
        with torch.no_grad():
            expected = zero(x)
            actual = phase(x)
        r = pt.corr(actual.numpy(), expected.numpy())
        d = pt.maxd(actual.numpy(), expected.numpy())
        print(f"phase vs zero-stuff PyTorch: corr {r:.8f} max|d| {d:.3e}")
        if r < 0.999999 or d > 2e-5:
            raise RuntimeError("phase-packed decoder failed PyTorch parity")

        stem = f"pt_mimi_deconly_w{width}_phase"
        fp32 = os.path.join(pt.OUT, stem + ".tflite")
        fp16 = os.path.join(pt.OUT, stem + "_fp16.tflite")
        pt.convert(phase, (torch.zeros(1, pt.MIMI_D, width),), fp32)
        pt.opcheck(fp32, stem)
        pt.to_fp16(fp32, fp16)
        pt.opcheck(fp16, stem + "_fp16")

        runner = pt.CM(fp16)
        with torch.no_grad():
            lite = runner(x.numpy())[0]
        r16 = pt.corr(lite, expected.numpy())
        d16 = pt.maxd(lite, expected.numpy())
        print(f"phase fp16 LiteRT CPU vs zero-stuff: corr {r16:.8f} max|d| {d16:.3e}")
        if r16 < 0.9999:
            raise RuntimeError("phase-packed fp16 graph failed audio-feature parity")


if __name__ == "__main__":
    main()
