# Tensor G5 FP16 half-truncation candidate

This branch records an opt-in AOT variant of the existing position-major FlowLM step. It changes only the Tensor G5 compiler's float truncation setting. The source graph, quantization, Android code, and production model selection are unchanged. The generated models remain under ignored `scripts/out/` in this worktree.

## Reproduction

The existing `pt_flowlm_fused_fp16_contiguous.tflite` was copied byte-for-byte from the parent checkout into this worktree's `scripts/out/`. In WSL, using the parent's pinned `.venv-aot` (`ai-edge-litert` and `ai-edge-litert-sdk-google-tensor` 2.2.0):

```bash
PT_OUT=/mnt/i/Android_Projects/PocketTTS-LiteRT/build/flowlm-worktrees/option9-g5-half-truncation/scripts/out \
  /mnt/i/Android_Projects/PocketTTS-LiteRT/.venv-aot/bin/python \
  scripts/aot_tensor_g5.py pt_flowlm_fused_fp16_contiguous --truncation half
```

The compiler accepted `half` for `Google_Tensor_G5`, took **33.3 s**, and reported `Subgraph 0 fully compiled: 717 / 717 ops offloaded to 1 partitions`. The existing no-truncation report also records 717/717 ops in one partition; its compile duration was not captured. No compiler retry was needed.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Shared position-major FP16 source | 169,288,096 | `1d6fe48ed4435b415970a240916a726f1b8994036aacf3c009085722a8a632c7` |
| Existing `no_truncation` G5 output | 172,399,184 | `1e3142667b944172d1216728bec52bfabe3529ac6606394dee2878c0945b8d08` |
| New `half` G5 output | 172,399,184 | `fb66bc91257207b809a4aab4a77e33168cd105cd4ee266405798a4937ccfdfac` |

The candidate file is `scripts/out/pt_flowlm_fused_fp16_contiguous_half_g5.tflite` in this worktree. The copied source hash matches the parent source. Both AOT outputs have the same byte length but different hashes; a direct byte comparison found 168,545,298 differing bytes. This establishes a distinct compiler output, but the opaque vendor program does not expose which internal instructions changed.

## Tensor interface and parity scope

Host LiteRT Interpreter inspection found identical public interfaces for both compiled outputs: one `serving_default` signature, seven float32 inputs, one float32 `[1,12321]` output, and one `DISPATCH_OP`. The K and V inputs remain position-major `[1,512,96,64]`; all other input shapes match the shared source graph. The export script checks exact eager equality of the position-major layout against the original fused step, then requires first- and next-step CPU fp32 graph max absolute differences below `2e-5`. Those checks precede the FP16 conversion; they establish layout and fp32 source parity, not FP16 or compiled-program parity. The FP16 AOT input is byte-identical between these two compilations. No numerical comparison between the compiled G5 programs is available on the WSL host. AOT `DISPATCH_OP` execution and latency require the Tensor G5 device.

## Pixel paired long workload

On Pixel 10 `frankel` (serial `57220DLCR002R6`, Android 17), the graph's on-device SHA-256 matched `fb66bc91257207b809a4aab4a77e33168cd105cd4ee266405798a4937ccfdfac`. The installed APK's `FlowLmHarnessTest.npuResidentCacheSpeechPair` ran once in each order with long text, Alba, seed 42, and `energyRepeats=1`. The candidate used logical graph `pt_flowlm_fused_fp16_contiguous_half.tflite`, `npuSliceCache=true`, `npuPositionMajorCache=true`, `npuResidentCache=false`, `verifyNpuSliceRows=false`, and `g5HighPerformance=false`. The control was the production dynamic INT8 FlowLM on CPU. Both orders reported `lm=NPU`, `dectx=NPU`, `dec=GPU` for the candidate and passed instrumentation. The APK was not reinstalled.

| Order | Frames NPU/CPU | Inference NPU/CPU | NPU/CPU | First audio NPU/CPU | LM run NPU/CPU | Waveform corr | Thermal |
|---|---:|---:|---:|---:|---:|---:|---:|
| NPU first | 750 / 759 | 32.221 / 24.844 s | 1.297x | 1.657 / 1.143 s | 16.454 / 15.043 s | 0.221575 | 0 |
| CPU first | 750 / 759 | 31.910 / 24.719 s | 1.291x | 1.657 / 1.128 s | 16.497 / 14.841 s | 0.221561 | 0 |
| Mean | 750 / 759 | 32.066 / 24.782 s | 1.294x | 1.657 / 1.136 s | 16.476 / 14.942 s | 0.221568 | 0 |

The half candidate stopped **nine frames (0.72 s) short** in both orders, whereas the prior no-truncation position-major FP16 pairs completed 759/759 frames. Its mean NPU/CPU inference ratio was 1.294x, versus 1.308x for the prior no-truncation pairs (32.682 / 24.992 s). The half run is shorter, so that small timing difference does not establish a speed gain. It misses both completion and CPU speed parity. The NPU LM graph-run stage alone remained 1.103x slower than CPU int8 on average.

The candidate's LM input/run/read times were 1.290/16.454/2.375 s and 1.229/16.497/2.369 s; the native contiguous cache row copies took 1.841 and 1.833 s. The NPU WAV was byte-identical across the two orders (SHA-256 `121b15bfe9161f3cf973c51c5b111c78499f59f3717fa9443f8198fd37617d0f`). Correlation to the longer CPU output is a diagnostic only. No listening or intelligibility assessment was performed.

The duration-adjusted audio-only-subtracted CPU/GPU/TPU energy totals were 16.706 and 16.666 J for the candidate, versus 29.399 and 29.677 J for CPU control. These one-repeat system monitor estimates do not establish a stable energy advantage. Thermal status was 0 in both runs.

Raw reports and WAVs are preserved under ignored `scripts/out/pixel-half-pair/npu-cpu/` and `scripts/out/pixel-half-pair/cpu-npu/` in this worktree. The corresponding device directories are `flowlm-npu-slice-position-major/speech-20261006-163624-136/` and `speech-20261006-164053-468/`. The candidate remains opt-in; production defaults were not changed.
