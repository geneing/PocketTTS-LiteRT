# Pixel 10 FlowLM OpenCL K/V cache experiment

Status: **opt-in implementation verified; not a performance or completion win.**
Branch `codex/flowlm-gpu-opencl-kv` keeps the original GPU profile checkpoint
(`b6f9974`) and the AHWB compatibility gate (`3d3dbbc`) in its ancestry.
The stable long-pair implementation is `89eb10b`.

## Protocol and implementation

`FlowLmHarnessTest#gpuSpeechPair` ran on Pixel 10 serial `57220DLCR002R6`,
Android 17 build `CP3A.260905.009`, Alba, seed 42, the harness's long repair-fair
text, in both GPU→CPU and CPU→GPU orders. Each arm ran the harness warmup,
long baseline, audio-only playback energy interval, then measured long take.
The CPU reference is `pt_flowlm_fused_dyn8_all.tflite` SHA-256
`895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9`.
Both arms used CPU decoder transformer and GPU SEANet. The GPU graph is the
position-major fused FP16 `pt_flowlm_fused_fp16_contiguous.tflite`, SHA-256
`1d6fe48ed4435b415970a240916a726f1b8994036aacf3c009085722a8a632c7`.
The installed device graph hash matched before testing.
App logcat recorded `717/717` nodes delegated to one `LITERT_CL` partition
for this position-major graph; the filtered transcript is retained at ignored
`scripts/out/gpu-flowlm/delegation-logcat.txt`.

The opt-in `gpuOpenClCache=true` path compiles with LiteRT 2.2.0's public C API.
K/V are persistent `OpenClBufferPacked` (type 14) buffers, seeded once per
utterance; the position-major graph exposes each new K/V row contiguously.
After `LiteRtRunCompiledModel`, a public LiteRT output read lock waits for GPU
completion. Two standard `clEnqueueCopyBuffer` calls on LiteRT's **borrowed
in-order queue** write the new rows at the current position; `clFinish` waits
before the next step. The host consumes 33 control floats, never the full K/V
bank per step. Strict graph layout checks require cache `[1,512,96,64]` and
output `[1,12321]`. Runtime CL allocations were 12,582,912 bytes per K/V bank
and 49,296 bytes for the packed output (49,284 useful bytes plus padding).
The AAR packages `libLiteRt.so` but no C headers; minimal version-pinned
public declarations are in the native bridge, resolved by public symbol names.

This applies the [Transformer-Lite paper's fixed-capacity, sub-tensor K/V
idea](https://arxiv.org/abs/2403.20041) to the interfaces available here:
the graph emits only new K/V, the two preallocated cache banks retain their
contents across steps, and row copies target byte offsets in those banks.
The existing fused graph packs control, K, and V into one output tensor, so it
cannot bind its K/V outputs directly to two offset buffer views without a
new graph export. The two same-queue GPU copies are the supported equivalent
for this candidate. The existing graph is fused and delegated in one partition;
this experiment does not claim additional kernel fusion or general GPU memory
reuse beyond those persistent cache buffers.

The first direct-queue attempt used an independent same-context OpenCL queue.
It read an all-zero first output and then the prior step's output: queue
ordering was insufficient. A long run on that version was explicitly stopped
and is **not** included below. The corrected borrowed-queue path passed the
three-token FP16 CPU parity gate, correlation `0.99999772`, mean absolute
difference `0.001367`, before the long paired runs.

## Paired long results

All times below are the measured take in seconds. “Read” includes output
completion and row-copy wait, so `run()` alone is not total GPU work.

| Order | Arm | Frames | Synthesis | First audio | LM input | LM run | LM read | LM steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPU→CPU | GPU FP16 cache | 754 | 87.072 | 4.497 | 5.810 | 2.187 | 52.227 | 1,102 |
| GPU→CPU | CPU dyn8 | 759 | 34.502 | 1.487 | 2.131 | 13.972 | 0.123 | 1,107 |
| CPU→GPU | GPU FP16 cache | 754 | 87.055 | 4.508 | 5.671 | 2.142 | 52.489 | 1,102 |
| CPU→GPU | CPU dyn8 | 759 | 34.523 | 1.472 | 2.165 | 13.956 | 0.123 | 1,107 |

GPU host input bytes were `267,892,352` in either order, versus CPU
`27,900,154,944`. GPU host output bytes were `145,464` versus CPU
`54,557,388`. Re-seeding once per sentence accounts for most GPU input bytes.
The GPU output lock alone cost 48.425/48.661 s; copy enqueue cost
0.095/0.085 s and copy wait 3.705/3.742 s. GPU Mimi decoder transformer cost
17.692/17.701 s, compared with CPU arm 13.002/12.991 s; SEANet was
4.596/4.563 s versus 4.485/4.461 s.

The GPU path is **2.52× slower** than the CPU reference in both orders, though
it improves the previous nonresident GPU measured take from 112.693 s to about
87.06 s under this protocol. It still finishes five frames earlier (60.32 s
audio versus 60.72 s). GPU-vs-CPU waveform correlation is `0.1601784`, lag
0, SNR `0.113 dB`, high-band error `0.265 dB`, HNR CPU/GPU `0.800/0.769 dB`.
These are diagnostics for different FP16/dyn8 free runs, not a listening
assessment or a claim that the GPU cache changed quality; the previous plain
GPU FP16 run had the same frame and waveform diagnostics. Thermal status was
0→0 in both orders. Duration-adjusted, audio-only-subtracted GPU/0 energy was
22.216/21.836 J for the GPU arm versus 4.543/4.528 J for CPU; overlapping
power domains must not be summed.

The full reports and both WAVs are retained locally, ignored by Git:
`scripts/out/gpu-flowlm/opencl-long-gpu-cpu/` and
`scripts/out/gpu-flowlm/opencl-long-cpu-gpu/`. The parity gate floats/reports
are under `scripts/out/gpu-flowlm/opencl-borrowed-pass/` and
`scripts/out/gpu-flowlm/opencl-gate/`.

## Reproduction and limits

Build `:app:assembleDebug :app:assembleDebugAndroidTest`, install both APKs,
and retain the model files in the app's external files directory. Run the
instrumentation runner directly; Gradle's connected test task uninstalls the
app and removes those models:

```text
adb -s 57220DLCR002R6 shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuOpenClPromptGate com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb -s 57220DLCR002R6 shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuSpeechPair -e gpuOpenClCache true -e gpuGraph pt_flowlm_fused_fp16_contiguous.tflite -e workload long -e order gpu-cpu -e voice alba -e seed 42 com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
```

Repeat the second line with `-e order cpu-gpu`. No production default changes.
The current runner still maps the whole 49 KB packed output through LiteRT's
read lock to synchronize. The user-requested direct 33-float OpenCL read on
LiteRT's borrowed queue is a separate follow-up candidate, as is the official
`benchmark_model --use_profiler=true` per-op diagnostic; neither result is
included in this paired baseline.
