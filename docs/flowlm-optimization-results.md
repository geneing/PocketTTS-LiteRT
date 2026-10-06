# Flow-LM accelerator experiments

Pixel 10 / Tensor G5, tested 2026-10-05 with the 201-word, 1,168-character
paragraph in `PowerBenchmarkTest`. Waveform correlation is the numerical audio
comparison; the WAV pairs below are for listening review. SNR is not used as a
quality gate.

## Results

| Flow-LM path | Runtime | RTF | Audio corr vs CPU | Incremental CPU/GPU/TPU energy* |
|---|---:|---:|---:|---:|
| CPU fp16 reference | 35.7 s for 60.2 s (fp16 test) | 1.69x | 1.000 | 48.3 J (fp16 test) |
| GPU fp16, one step | 97.0 s for 60.3 s | 0.62x | 0.193 | 29.4 J |
| CPU dynamic-range int8 reference | 24.7 s for 60.7 s | 2.46x | 1.000 | 26.9 J (int8 test) |
| GPU dynamic-range int8, one step | 93.4 s for 60.2 s | 0.64x | 0.112 | 22.7 J |
| CPU fp16 reference (NPU test) | 36.4 s for 60.2 s | 1.65x | 1.000 | 52.9 J (NPU test) |
| Tensor G5 NPU fp16, no truncation | 48.7 s for 60.7 s | 1.25x | 0.085 | 16.6 J |

*Energy is the sum of the CPU, GPU, and TPU monitor deltas for synthesize-plus-
playback minus duration-scaled audio-only playback. It is a device-level
estimate, and individual monitor deltas can be noisy. Paired rows use the same
decoder placement so the Flow-LM placement is the main changed component.

The GPU fp16 step spent about 25.8 s staging inputs and 54.1 s reading results;
the GPU int8 step spent about 27.2 s staging and 53.1 s reading. The 25 MB KV
cache transfer dominates both. Int8 weights did not remove that cost. Both GPU
paths were confirmed to run on GPU, with no CPU fallback.

The NPU fused graph's warm one-step invocation was 14.6–15.6 ms, versus
21.2–21.3 ms on CPU fp16; the first NPU invocation took about 92 ms. End-to-end
paragraph RTF remained slower than CPU. The NPU AOT export with
`--truncation no_truncation` was byte-identical to the SDK-default export
(SHA-256 `0d6eba65d634a9e9f9416f85f1b72fd01baae00ebae64d9ef7f1ec1adc3e19f1`),
and both produced the same long-clip WAV. The precision option therefore did
not change this graph. Single-step tensor cosines were high (about 0.9998 to
0.99999), but the long free-running output diverged substantially.

The NPU energy estimate was lower, but its RTF remains below the CPU fp16
reference. GPU fp16 and int8 RTF are substantially below their CPU references.

## Listening review

On 2026-10-05, the user listened to the NPU, GPU fp16, and GPU dynamic-range
int8 candidates and said all three sound fine. Their low waveform correlations
are retained as diagnostics, not quality rejection gates. The Flow-LM power
tests log correlation and sample lengths while same-engine repeatability checks
remain gated at 0.99.

## Listening files

These WAVs are generated locally under the ignored `scripts/out/` directory:

- NPU test pair: `flowlm_npu_power_reference.wav` and
  `flowlm_npu_power_candidate.wav`.
- GPU fp16 pair: `flowlm-gpu-fp16-step-reference.wav` and
  `flowlm-gpu-fp16-step-candidate.wav`.
- GPU int8 pair: `flowlm-gpu-dyn8-int8-reference.wav` and
  `flowlm-gpu-dyn8-int8-candidate.wav`.

## Pending

The N=4 fp16 GPU power test was stopped before it produced measurements when
the user needed the phone. Resume it in isolation when the phone is available.
It is the remaining GPU experiment that reduces per-step host/device round trips.

The focused diagnostic is `FlowLmNpuPrecisionTest`; the paired power probes are
in `PowerBenchmarkTest`. Run one instrumentation method at a time. Do not run
`:app:connectedDebugAndroidTest`, which uninstalls the app and removes the
pushed models.
