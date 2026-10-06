#!/usr/bin/env python3
"""Build the first gate for a resident Tensor G5 FlowLM KV cache.

Run in the existing Linux conversion environment, then AOT environment:

  python scripts/probe_npu_cache_chain.py export
  python scripts/probe_npu_cache_chain.py aot

The graph deliberately has the same 25,165,824-byte fp32 cache footprint as
the two packed FlowLM K/V inputs. Only the first cache row is updated; the
scalar output makes a three-step chained run observable without reading cache.
The Android FlowLmHarnessTest.probeNpuCacheChain method exercises the AOT file.
"""

import argparse
import os
from pathlib import Path


CACHE_SHAPE = (2, 96, 512, 64)
ROW_SHAPE = (2, 96, 1, 64)
SOURCE = "pt_npu_cache_chain.tflite"
COMPILED = "pt_npu_cache_chain_g5.tflite"


def export(out: Path) -> None:
    import torch
    from torch import nn
    import litert_torch

    class CacheUpdate(nn.Module):
        def forward(self, cache, row):
            first = cache[:, :, :1, :] + row
            next_cache = torch.cat((first, cache[:, :, 1:, :]), dim=2)
            scalar = first[0, 0, 0, 0].reshape(1)
            return next_cache, scalar

    torch.manual_seed(0)
    model = CacheUpdate().eval()
    cache = torch.zeros(CACHE_SHAPE, dtype=torch.float32)
    row = torch.zeros(ROW_SHAPE, dtype=torch.float32)
    row[0, 0, 0, 0] = 1
    with torch.no_grad():
        next_cache, scalar = model(cache, row)
    assert tuple(next_cache.shape) == CACHE_SHAPE and scalar.item() == 1
    dst = out / SOURCE
    litert_torch.convert(model, (cache, row)).export(str(dst))
    print(f"exported {dst} ({dst.stat().st_size} bytes); cache={cache.numel() * 4} bytes")


def aot(out: Path) -> None:
    from ai_edge_litert.aot.vendors.google_tensor import target as gt
    from aot_tensor_g5 import aot_one

    src = out / SOURCE
    dst = out / COMPILED
    if not src.is_file():
        raise SystemExit(f"missing {src}; run export first")
    work = out / "_aot_cache_chain"
    work.mkdir(exist_ok=True)
    elapsed, report = aot_one(str(src), str(dst),
                              gt.Target(gt.SocModel.TENSOR_G5), str(work))
    print(f"AOT {elapsed:.1f}s: {report}\n{dst} ({dst.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "aot"))
    parser.add_argument("--out", type=Path,
                        default=Path(os.environ.get("PT_OUT", Path(__file__).parent / "out")))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "export":
        export(args.out)
    else:
        aot(args.out)


if __name__ == "__main__":
    main()
