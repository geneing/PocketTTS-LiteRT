# Pixel 10 fused FlowLM GPU profile

Status: **one completed GPU-first long pair; reversed pair stopped at user request**.
The GPU path is an opt-in harness experiment. No production placement or graph
default changed.

## Setup and dispatch

- Device: Pixel 10 (`frankel`), Android 17,
  `google/frankel/frankel:17/CP3A.260905.009/16091614:user/release-keys`;
  serial `57220DLCR002R6`.
- Candidate: `pt_flowlm_fused_fp16.tflite`, SHA-256
  `1cf47aa0668bb4c6238c869db8902682377727addc0cb8d8f938b63a2a0f2b01`.
  Reference: shipped CPU `pt_flowlm_fused_dyn8_all.tflite`, SHA-256
  `895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9`.
  Device hashes matched the local artifacts.
- The two speech arms use the same Mimi placement: decoder transformer on CPU,
  SEANet on GPU. This avoids process-global NPU dispatch setup after GPU LM
  load. Both use Alba, noise seed 42, and the long paragraph in
  `FlowLmHarnessTest`. The GPU arm uses the existing optional LiteRT program
  cache directory.
- LiteRT registered its OpenCL accelerator and delegated **715/715 fused FP16
  FlowLM nodes to one `LITERT_CL` partition**. The harness reported
  `lm=GPU`, `dectx=CPU`, `dec=GPU`, `dec_w=GPU`. The CPU dyn8 FlowLM delegated
  666/668 step nodes to XNNPACK in three partitions. The GPU cache API probe
  from option 4 remains a separate, unpassed residency gate; this run uses the
  ordinary host K/V buffers.

## Prompt smoke and stage timing

`runTextPromptHarness` completed with 14 Alba prompt tokens. GPU FP16 output
versus CPU dyn8 had correlation **0.99994123** and mean absolute difference
**0.00640878**. Times below sum across the 14 steps in milliseconds:

| Backend | Model load | K/V input writes | `run()` | Output read/sync | Host K/V row copy | First `run()` |
|---|---:|---:|---:|---:|---:|---:|
| GPU FP16 | 1,081.76 | 272.71 | 35.55 | 739.33 | 16.97 | 0.97 |
| CPU dyn8 | 1,774.96 | 31.03 | 158.45 | 1.51 | 1.34 | 14.45 |

The GPU `run()` time is mainly dispatch: output `readFloat()` waits for GPU
work and copies the result. Counting `run()` alone would reverse the result.
The complete GPU prompt path takes about **1,065 ms** across these four stages,
versus **192 ms** for CPU. The fused graph requires both full packed K/V banks
as inputs each step (about 25.2 MB), so the JVM path still copies them on every
invocation. The GPU delegate accepted the graph; host transfer and readback are
the measured blockers.

## Complete long pair: GPU then CPU

`FlowLmHarnessTest#gpuSpeechPair`, order `gpu-cpu`, passed on device. Each arm
had a short warmup, one same-workload baseline, an audio-only playback interval,
and one measured synthesis. Timings below are the measured take. PowerMonitor
values use duration-scaled audio-only subtraction; rail domains overlap and
must not be added together.

| Measure | GPU FP16 | CPU dyn8 |
|---|---:|---:|
| Synthesis | 112.693 s | 35.126 s |
| Output | 754 frames / 60.32 s | 759 frames / 60.72 s |
| First audio | 5.941 s | 1.529 s |
| LM input writes | 26.832 s | 2.266 s |
| LM `run()` | 3.261 s | 14.257 s |
| LM read/sync | 56.310 s | 0.128 s |
| LM steps | 1,102 | 1,107 |
| LM host input bytes | 27,774,137,984 | 27,900,154,944 |
| Mimi decoder transformer | 17.331 s | 13.275 s |
| Mimi SEANet | 4.214 s | 4.399 s |
| Incremental GPU/0 energy | 22.823 J | 4.416 J |
| Incremental CPU/0 energy | 6.963 J | 14.930 J |

GPU synthesis was **3.21× slower** than CPU. Its LM input and read stages
alone took 83.142 s; the `run()` stage accounts for just 3.261 s. Both arms
reported thermal status 0 at the start and end. The GPU arm ended five frames
early. Waveform correlation was 0.16018 (lag 0), SNR 0.113 dB, high-band error
0.265 dB, and HNR 0.800/0.768 dB for CPU/GPU. These are diagnostics, not a
listening or intelligibility judgment; the frame deficit already fails exact
completion against this reference.

The full report, both WAVs, instrumentation transcript, and filtered LiteRT
logcat were pulled to ignored
`scripts/out/gpu-flowlm/pair-gpu-cpu/`. The prompt report and raw float
tensors are in `scripts/out/gpu-flowlm/smoke-fp16/`.

## Reversed order and pause state

The `cpu-gpu` instrumentation was started and **terminated at the user's pause
request**. Its logcat shows the CPU warmup (23 frames) and CPU long baseline
(759 frames, 35.229 s), then stopped before the measured CPU take. It never
reached the GPU arm. The new device run
directory was empty, with no report or WAV, when both app processes were
force-stopped. The partial instrumentation transcript and filtered logcat are
preserved in ignored `scripts/out/gpu-flowlm/pair-cpu-gpu/`.

The completed order is enough to reject this JVM GPU path for a speed or exact
completion claim, but it is **not** a completed two-order paired result. No
further GPU variants or Pixel tests were run after the pause.

## Reproduction

Build and install the opt-in harness from this branch, keeping the existing
model files in the app's external files directory. Invoke the runner directly
to preserve those files (the Gradle connected test task uninstalls the app):

```text
adb -s 57220DLCR002R6 shell "am instrument -w -e class com.pockettts.FlowLmHarnessTest#runTextPromptHarness -e backends GPU -e lmGraph pt_flowlm_fused_fp16.tflite -e referenceGraph pt_flowlm_fused_dyn8_all.tflite -e requireCandidates true com.pockettts.test/androidx.test.runner.AndroidJUnitRunner"
adb -s 57220DLCR002R6 shell "am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuSpeechPair -e order gpu-cpu -e workload long -e voice alba -e seed 42 -e energyRepeats 1 com.pockettts.test/androidx.test.runner.AndroidJUnitRunner"
```

The reverse order changes `-e order` to `cpu-gpu`. Reports and WAVs appear
under the app's `flowlm-harness/` and `flowlm-gpu/` external files directories.
