#!/usr/bin/env python3
"""AOT-compile the Pocket TTS graphs for a Google Tensor NPU.

The Tensor NPU does not JIT anything at runtime: LiteRT's Google Tensor
dispatch shim only executes a graph that was *ahead-of-time* compiled into a
Tensor partition by this plugin. So an NPU variant of a graph is a different
file, not a different accelerator flag -- `<graph>_g6.tflite`, holding a
dispatch partition in place of the original ops.

This uses the `ai_edge_litert.aot` API, not `litert_torch`, and therefore needs
its own environment: `ai-edge-litert==2.3.0` +
`ai-edge-litert-sdk-google-tensor==2.3.0` (the vendor backend that registers
`SocModel.TENSOR_G6`). The conversion venv's `ai-edge-litert==2.1.6` /
`litert-torch` stack is unrelated and untouched.

The 2.3.0 SDK no longer downloads the compiler at install time. Point
`GOOGLE_TENSOR_SDK_BETA` at the `litert_plugin_compiler.tar.gz` bundle (the
notebook's `%env` does the same) before installing the SDK package:

    GOOGLE_TENSOR_SDK_BETA=$PWD/litert_plugin_compiler.tar.gz \
        uv pip install "ai-edge-litert==2.3.0" \
                       "ai-edge-litert-sdk-google-tensor==2.3.0"

The app's Android runtime must be the same LiteRT release: `litert:2.3.0`,
`litert-gpu:2.3.0`, and `litert-npu-runtime-google-tensor:2.3.0`, which now
ships the dispatch shim in its AAR.

Run:  python aot_tensor_npu.py [graph ...]
      (default: every existing source graph that is worth compiling)
      --soc g6            target SoC (default; g5 is kept for the old phone)
      PT_OUT=<dir>  where the source .tflite files are read from and the
                    `*_g6.tflite` files are written (default scripts/out/)

What compiles and what does not (measured, Tensor SDK 2.2.0 / G5):

  pt_flowlm_fused_fp16   715 / 715 ops, 1 partition    ~29 s
  pt_flowlm_ms4_fp16    2785 / 2785 ops, 1 partition    ~95 s
  pt_mimi_dec_tx_fp16    210 / 210 ops, 1 partition     ~9 s
  pt_mimi_deconly_fp16   FAILED on 2.2.0/G5: all ops partition, then the
                         backend compiler errors out (INTERNAL). Same for the
                         fp32 graph, so it is the SEANet body (x16
                         ConvTranspose upsample), not fp16. Keep the SEANet
                         decoder on GPU/CPU.

Tensor SDK 2.3.0 / G6 (Pixel 11), for the record:

  pt_flowlm_fused_dyn8_all_contiguous  OK only with --truncation half
                                       (17 s, 670 ops); no_truncation crashes
  pt_flowlm_fused_dyn8_all             OK (20 min, 7147 + 668 ops)
  pt_flowlm_step_fp16 / head_fp16      OK
  pt_mimi_dec_tx_fp16                  FAILS (SIGABRT in the compiler worker)
  pt_flowlm_fused_fp16_contiguous      FAILS at every truncation setting

No 2.3.0-compiled AOT file loads on the public 2.3.x runtime; Pixel 11 runs the
FlowLM through on-device JIT instead (see docs/pixel11-g6.md).

fp16 is fine: the plugin accepts half-precision weights, and every graph above
compiles to byte-identical size in fp32/fp16 apart from the weight width.
"""
import argparse
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PT_OUT", os.path.join(HERE, "out"))

DEFAULT_GRAPHS = [
    "pt_flowlm_fused_fp16_contiguous",
    "pt_flowlm_fused_fp16",
    "pt_flowlm_ms4_fp16",
    "pt_mimi_dec_tx_fp16",
]

SOC = {
    "g5": ("TENSOR_G5", "Google_Tensor_G5"),
    "g6": ("TENSOR_G6", "Google_Tensor_G6"),
}


def aot_one(src, dst, target, export_name, work_dir, truncation=None):
    from ai_edge_litert.aot import aot_compile as aot_lib

    t0 = time.time()
    options = ({"google_tensor_truncation_type": truncation}
               if truncation is not None else {})
    result = aot_lib.aot_compile(src, output_dir=work_dir, target=[target],
                                 keep_going=False, **options)
    report = result.compilation_report().strip().replace("\n", " | ")
    result.export(work_dir, model_name="m")
    shutil.copy(os.path.join(work_dir, f"m_{export_name}.tflite"), dst)
    return time.time() - t0, report


def main():
    from ai_edge_litert.aot.vendors.google_tensor import target as gt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="*", help="source graph stems (without .tflite)")
    parser.add_argument("--soc", choices=tuple(SOC), default="g6",
                        help="Google Tensor SoC to compile for (default g6 / Pixel 11)")
    parser.add_argument("--truncation", choices=("half", "bfloat16", "no_truncation"),
                        help="Google Tensor compiler truncation setting; omission keeps the SDK default")
    args = parser.parse_args()
    graphs = args.graphs or DEFAULT_GRAPHS
    enum_name, export_name = SOC[args.soc]
    target = gt.Target(getattr(gt.SocModel, enum_name))
    suffix = f"_{args.soc}" if args.truncation is None else f"_{args.truncation}_{args.soc}"
    work_dir = os.path.join(OUT, f"_aot_work{suffix}")
    os.makedirs(work_dir, exist_ok=True)

    print(f"target {target!r} soc={args.soc} "
          f"truncation={args.truncation or 'sdk_default'} -> {OUT}")
    failed = []
    for g in graphs:
        src = os.path.join(OUT, f"{g}.tflite")
        dst = os.path.join(OUT, f"{g}{suffix}.tflite")
        if not os.path.exists(src):
            print(f"SKIP {g}: no {src}")
            continue
        try:
            secs, report = aot_one(src, dst, target, export_name, work_dir,
                                   args.truncation)
            print(f"OK   {g:44s} {secs:6.1f}s  {report}  "
                  f"-> {os.path.getsize(dst)/1e6:.1f} MB")
        except Exception as e:  # noqa: BLE001 - a failure here is a result
            failed.append(g)
            print(f"FAIL {g:44s}       {type(e).__name__}: "
                  f"{str(e).splitlines()[0]}")
    if failed:
        print(f"\nfailed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
