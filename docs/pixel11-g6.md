# Pixel 11 / Tensor G6 conversion notes

Target: Pixel 11 (`cubs`), Google Tensor G6, Android 17
(`CD1A.261005.003`), PowerVR C-Series CXTP-48-1536.

Compiled with the user-supplied `litert_plugin_compiler.tar.gz` (the
`ai-edge-litert-sdk-google-tensor==2.3.0` bundle) in
`~/pockettts-aot-g6/.venv`; the runtime is LiteRT 2.3.2
(`litert`, `litert-gpu`, `litert-npu-runtime-google-tensor`).

## What compiled for G6

| graph | result |
|---|---|
| `pt_flowlm_fused_dyn8_all_contiguous` (dynamic int8, position-major, float I/O) | **OK** with `--truncation half` (17 s, 89.1 MB, 670/670 ops, 1 partition); `no_truncation` fails |
| `pt_flowlm_fused_dyn8_all` (group-major, two signatures) | OK (20 min, 216.8 MB, 7147 + 668 ops) |
| `pt_flowlm_step_fp16` | OK (17.6 s, 153.1 MB, 526 ops) |
| `pt_flow_head_fp16` | OK (3.7 s, 20.0 MB, 187 ops) |
| `pt_mimi_dec_tx_fp16` / fp32 | **FAILS**: all 210 ops partition, then the compiler worker aborts (SIGABRT) for G6. Same for all truncation settings and fp32. It still compiles for G5 with 2.3.0 |
| `pt_mimi_deconly_fp16` | fails as before (SEANet body) |
| `pt_flowlm_fused_fp16_contiguous` | fails on G6 with or without truncation |

## The AOT files do not load on the 2.3.x runtime

- With `litert:2.3.0` + `litert-npu-runtime-google-tensor:2.3.0`, the dispatch
  shim reports `Accelerator returned null hooks` at
  `LiteRtCreateCompiledModel`, before the graph matters.
- With `litert:2.3.2` + the 2.3.2 NPU runtime, the shim initializes but the
  2.3.0-compiled partition fails with `Failed to get SB invocation context`
  and `Encountered unresolved custom op: DISPATCH_OP`.
- There is no public 2.3.2 compiler SDK on PyPI (`ai-edge-litert-sdk-google-tensor`
  stops at 2.3.0), so there is no matching compiler/runtime pair for G6 yet.

## Working NPU path: on-device JIT

The 2.3.2 NPU runtime package also contains
`libLiteRtCompilerPlugin_google_tensor.so`. Loading the **uncompiled**
position-major dynamic-int8 source with `Accelerator.NPU` compiles it on the
device and caches the result
(`NPU JIT compilation caching enabled with cache dir: .../cache`). The app
ships it as `pt_flowlm_fused_dyn8_all_contiguous_jit.tflite`.

Measured on the phone (Alba, seed 42, dectx CPU, SEANet GPU):

| workload | LM placement | audio | wall | first audio | LM stage |
|---|---|---:|---:|---:|---:|
| short, 18 frames | NPU JIT | 1.44 s | 0.65 s inference | 0.65 s | 0.376 s |
| short, 18 frames | CPU int8 | 1.44 s | 0.59 s inference | 0.59 s | 0.280 s |
| 163–178 frames | NPU JIT | 13.0 s | 16.5 s | 3.74 s | 7.69 s |
| 163–178 frames | CPU int8 | 14.2 s | 16.4 s | 3.26 s | 6.43 s |

NPU output matches CPU int8 almost exactly (waveform corr 0.9999999999) because
the JIT compiles the same dynamic-int8 graph. It is **not faster** on G6, and
`dec_tx` cannot move to the NPU because the compiler aborts on that graph. The
default is therefore CPU LM + CPU dec_tx + GPU SEANet; the NPU is opt-in.

## Opting into the NPU

```bash
adb shell "echo lm > /sdcard/Android/data/com.pockettts/files/force_npu.txt"
# relaunch the app, or run the harness
adb shell am instrument -w \
  -e class com.pockettts.FlowLmHarnessTest#npuResidentCacheSpeechPair \
  -e workload short -e npuSliceCache true -e npuPositionMajorCache true \
  -e npuGraph pt_flowlm_fused_dyn8_all_contiguous_jit.tflite \
  com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
```

Remove `force_npu.txt` (or write only `dectx` when a G6 dec_tx graph exists) to
return to the CPU default.

The native `g5HighPerformance` probe reaches `LiteRtCreateCompiledModel` on
2.3.2 but the HIGH_PERFORMANCE compile fails with status 504 on G6, so that flag
stays off. The slice-cache bridge itself works (buffer types report host memory,
not AHWB, on G6).

## AOT reproduction

```bash
# source graph (conversion venv)
PT_OUT=scripts/out python scripts/export_flowlm_npu_contiguous.py
PT_OUT=scripts/out python scripts/quantize_flowlm_g5.py --recipe dynamic8

# G6 AOT (AOT venv built from the supplied tarball)
PT_OUT=scripts/out python scripts/aot_tensor_npu.py \
    pt_flowlm_fused_dyn8_all_contiguous --soc g6 --truncation half
```

The AOT file is kept as `..._half_g6.tflite` for the record and for a future
compiler/runtime pair; the app does not load it.
