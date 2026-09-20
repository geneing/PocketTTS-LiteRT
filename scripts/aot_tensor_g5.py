#!/usr/bin/env python3
"""AOT-compile the Pocket TTS graphs for the Google Tensor G5 NPU (Pixel 10).

The Tensor NPU does not JIT anything at runtime: LiteRT's Google Tensor
dispatch shim only executes a graph that was *ahead-of-time* compiled into a
Tensor partition by this plugin. So an NPU variant of a graph is a different
file, not a different accelerator flag -- `<graph>_g5.tflite`, holding a
dispatch partition in place of the original ops.

This uses the `ai_edge_litert.aot` API, not `litert_torch`, and therefore needs
its own environment: `ai-edge-litert==2.2.0` +
`ai-edge-litert-sdk-google-tensor==2.2.0` (the vendor backend that registers
`SocModel.TENSOR_G5`). The conversion venv's `ai-edge-litert==2.1.6` /
`litert-torch` stack is unrelated and untouched. The SDK package downloads and
bundles the compiler plugin (`liblitert_plugin_compiler.so`, ~164 MB) plus the
darwinn RISC-V toolchain at install time, so no `GOOGLE_TENSOR_SDK_BETA`
tarball is needed afterwards.

Run:  python aot_tensor_g5.py [graph ...]
      (default: flowlm_fused + flowlm_ms4 + mimi_dec_tx, fp16)
      PT_OUT=<dir>  where the source .tflite files are read from and the
                    `*_g5.tflite` files are written (default scripts/out/)

What compiles and what does not (measured, Tensor SDK 2.2.0):

  pt_flowlm_fused_fp16   715 / 715 ops, 1 partition    ~29 s
  pt_flowlm_ms4_fp16    2785 / 2785 ops, 1 partition    ~95 s
  pt_mimi_dec_tx_fp16    210 / 210 ops, 1 partition     ~9 s
  pt_mimi_deconly_fp16   FAILS: all ops partition, then the backend compiler
                         errors out (INTERNAL). Same for the fp32 graph, so it
                         is the SEANet body (x16 ConvTranspose upsample), not
                         fp16. Keep the SEANet decoder on GPU/CPU.

fp16 is fine: the plugin accepts half-precision weights, and every graph above
compiles to byte-identical size in fp32/fp16 apart from the weight width.
"""
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PT_OUT", os.path.join(HERE, "out"))

DEFAULT_GRAPHS = [
    "pt_flowlm_fused_fp16",
    "pt_flowlm_ms4_fp16",
    "pt_mimi_dec_tx_fp16",
]


def aot_one(src, dst, target, work_dir):
    from ai_edge_litert.aot import aot_compile as aot_lib

    t0 = time.time()
    result = aot_lib.aot_compile(src, output_dir=work_dir, target=[target],
                                 keep_going=False)
    report = result.compilation_report().strip().replace("\n", " | ")
    result.export(work_dir, model_name="m")
    shutil.copy(os.path.join(work_dir, "m_Google_Tensor_G5.tflite"), dst)
    return time.time() - t0, report


def main():
    from ai_edge_litert.aot.vendors.google_tensor import target as gt

    graphs = sys.argv[1:] or DEFAULT_GRAPHS
    target = gt.Target(gt.SocModel.TENSOR_G5)
    work_dir = os.path.join(OUT, "_aot_work")
    os.makedirs(work_dir, exist_ok=True)

    print(f"target {target!r} -> {OUT}")
    failed = []
    for g in graphs:
        src = os.path.join(OUT, f"{g}.tflite")
        dst = os.path.join(OUT, f"{g}_g5.tflite")
        if not os.path.exists(src):
            print(f"SKIP {g}: no {src}")
            continue
        try:
            secs, report = aot_one(src, dst, target, work_dir)
            print(f"OK   {g:28s} {secs:6.1f}s  {report}  "
                  f"-> {os.path.getsize(dst)/1e6:.1f} MB")
        except Exception as e:  # noqa: BLE001 - a failure here is a result
            failed.append(g)
            print(f"FAIL {g:28s}       {type(e).__name__}: "
                  f"{str(e).splitlines()[0]}")
    if failed:
        print(f"\nfailed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
