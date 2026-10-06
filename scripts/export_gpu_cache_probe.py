#!/usr/bin/env python3
"""Export the fixed-shape GPU cache-chain gate without loading Pocket TTS.

Run in the Linux conversion environment documented in AGENTS.md:
    PT_OUT=scripts/out python scripts/export_gpu_cache_probe.py

The graph models the combined fp32 K/V cache (2 * 96 heads * 512 * 64). A
one-hot position mask and one K/V row update exactly one cache position. The
scalar output gives the host a small synchronization/readback target. This
probe says nothing about FlowLM quality or actual GPU residency by itself.
"""

import os
from pathlib import Path

import torch
from torch import nn


CHANNELS = 2 * 96
PMAX = 512
HD = 64


class CacheUpdate(nn.Module):
    def forward(self, cache, row, position_mask):
        next_cache = cache + row * position_mask
        # The host reads only one float per step. The full cache must remain an
        # output because it becomes the following invocation's input.
        probe = row[:, :1, :, :1].reshape(1, 1)
        return next_cache, probe


def main():
    import litert_torch

    output_dir = Path(os.environ.get("PT_OUT", Path(__file__).resolve().parent / "out"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "pt_gpu_cache_probe.tflite"
    example = (
        torch.zeros(1, CHANNELS, PMAX, HD),
        torch.zeros(1, CHANNELS, 1, HD),
        torch.zeros(1, 1, PMAX, 1),
    )
    litert_torch.convert(CacheUpdate().eval(), example).export(str(output))
    print(f"exported {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
