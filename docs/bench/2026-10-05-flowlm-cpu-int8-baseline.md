# Pixel 10 FlowLM CPU int8 baseline

Measured 2026-10-05 on Pixel 10 (`frankel`), Android 17 build
`google/frankel/frankel:17/CP3A.260905.009/16091614:user/release-keys`.
The model was `pt_flowlm_fused_dyn8_all.tflite` (SHA-256
`895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9`), with
LiteRT Android runtime 2.2.0. The flow harness uses zero noise and compares the
CPU int8 graph against itself. Prompt rows include voice prefix state; both
`alba` and `marius` have 126 prefix positions.

## Focused FlowLM prompt harness

Ran `FlowLmHarnessTest#runTextPromptHarness` directly with CPU placement,
`lmGraph=pt_flowlm_fused_dyn8_all.tflite`, and the same graph as
`referenceGraph`. `runMs` is the sum of `CompiledModel.run` calls; it excludes
model load, host input writes, output reads, and any audio decode. The candidate
and reference each load a separate model instance.

| Prompt regime | Voice | Tokens | Reference run (ms) | Candidate run (ms) | Output corr | Mean abs diff |
|---|---|---:|---:|---:|---:|---:|
| Short (`Hello there.`) | alba | 3 | 37.61 | 47.29 | 1.00000000 | 0 |
| Short (`Hello there.`) | marius | 3 | 37.84 | 44.51 | 1.00000000 | 0 |
| Medium (one garden sentence) | alba | 35 | 406.79 | 407.60 | 1.00000000 | 0 |
| Medium (one garden sentence) | marius | 35 | 396.84 | 405.19 | 1.00000000 | 0 |
| Near capacity (traveler sentence repeated 8 times) | alba | 336 | 3,924.04 | 3,887.05 | 1.00000000 | 0 |
| Near capacity (traveler sentence repeated 8 times) | marius | 336 | 3,877.74 | 3,929.67 | 1.00000000 | 0 |

Each model instance took about 1.72–1.75 s to load. The 336-token prompts fit
within the 512-position context with the 126-position voice prefix. An initial
12-repeat probe encoded to 504 tokens and was correctly rejected before graph
execution by the harness capacity check.

## Full-pipeline CPU control

Ran `PowerBenchmarkTest#longParagraphPlaybackPower` as a single instrumentation
method. This is the documented 201-word / 1,168-character paragraph with the
default placement (`lm:CPU`, `dectx:NPU`, `dec:GPU`). The test compares repeated
default synthesis and includes an audio-only playback interval for the energy
estimate.

| Audio duration | Synthesis time | RTF | Frames | Repeat audio corr | Incremental CPU + GPU + TPU energy |
|---:|---:|---:|---:|---:|---:|
| 60.72 s | 26.653 s | 2.278x | 759 | 1.000 | 28.314 J |

The measured model-stage time was 22.390 s: FlowLM 15.757 s, decoder transformer
2.073 s, and SEANet 4.560 s. Incremental energy is the sum of the CPU/0,
CPU/1, CPU/2, GPU/0, and TPU/1 deltas after subtracting the duration-scaled
audio-only interval. It is a device-level estimate, not a per-model rail.

## Coverage and limits

This establishes the CPU int8 control for short, medium, and near-capacity
prompts across two voices, plus one long-audio whole-pipeline control. The
focused harness tests FlowLM tensors only; the power test covers the long audio
regime. Option candidates still need paired full-pipeline and listening checks
when their focused correctness gates pass. The harness timing is not an
end-to-end latency measurement.
