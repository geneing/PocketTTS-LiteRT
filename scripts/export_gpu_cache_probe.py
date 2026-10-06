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

import numpy as np
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


def check_export(path):
    """Fail early if converter changes the positional I/O protocol."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    expected_input_shapes = [(1, CHANNELS, PMAX, HD), (1, CHANNELS, 1, HD),
                             (1, 1, PMAX, 1)]
    expected_output_shapes = [(1, CHANNELS, PMAX, HD), (1, 1)]
    actual_input_shapes = [tuple(d["shape"]) for d in inputs]
    actual_output_shapes = [tuple(d["shape"]) for d in outputs]
    assert actual_input_shapes == expected_input_shapes, actual_input_shapes
    assert actual_output_shapes == expected_output_shapes, actual_output_shapes

    cache = np.zeros(expected_input_shapes[0], dtype=np.float32)
    row = np.full(expected_input_shapes[1], 0.25, dtype=np.float32)
    mask = np.zeros(expected_input_shapes[2], dtype=np.float32)
    mask[0, 0, 7, 0] = 1.0
    for detail, data in zip(inputs, (cache, row, mask)):
        interpreter.set_tensor(detail["index"], data)
    interpreter.invoke()
    next_cache = interpreter.get_tensor(outputs[0]["index"])
    probe = interpreter.get_tensor(outputs[1]["index"])
    assert np.allclose(next_cache[:, :, 7, :], 0.25), "cache row update differs"
    assert np.count_nonzero(next_cache[:, :, :7, :]) == 0, "cache prefix changed"
    assert np.count_nonzero(next_cache[:, :, 8:, :]) == 0, "cache suffix changed"
    assert np.allclose(probe, 0.25), probe
    print("CPU TFLite parity: cache row and scalar exact; positional I/O verified")


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
    check_export(output)
    print(f"exported {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
