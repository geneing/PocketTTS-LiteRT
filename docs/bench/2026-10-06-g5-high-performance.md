# Pixel 10 Google Tensor HIGH_PERFORMANCE probe

## Result

The LiteRT 2.2.0 native C API accepted an opaque `google_tensor` TOML payload
containing `performance_mode = 3`, created a compiled NPU model, and ran it
against the Kotlin model's TensorBuffers. A one-token Alba prompt returned an
NPU output with correlation 0.99998886 to the CPU FP16 reference. The native
path then completed two long Alba speech pairs with the existing position-major
persistent AHWB cache. Both NPU and CPU arms produced 759 frames / 60.72 s.

| Order | HIGH_PERFORMANCE NPU | CPU int8 | NPU / CPU | First audio NPU / CPU | NPU LM input / run / read | NPU cache row copy | Thermal |
|---|---:|---:|---:|---:|---:|---:|---:|
| NPU then CPU | 30.247 s | 24.959 s | 1.212x | 1.573 / 1.126 s | 1.314 / 13.986 / 2.371 s | 1.841 s | 0 |
| CPU then NPU | 29.407 s | 25.144 s | 1.170x | 1.475 / 1.147 s | 1.230 / 13.627 / 2.191 s | 1.661 s | 0 |
| Mean | 29.827 s | 25.052 s | 1.191x | 1.524 / 1.137 s | 1.272 / 13.807 / 2.281 s | 1.751 s | 0 |

The prior default-mode position-major pairs averaged 32.682 s NPU and
24.992 s CPU (1.308x) on this same AOT graph and workload. The experimental
pairs are about 2.855 s faster on NPU, but still 4.776 s behind CPU on average.
The NPU graph-run stage is faster than CPU's (13.807 vs 15.211 s), while the
candidate's Mimi decoder transformer also takes about 3.63 s versus 1.64 s
in the CPU-LM arm, despite both arms selecting NPU for that graph. The harness
measures those stages but does not identify the cause of that difference.
The cache row copy is the `rowCopy` component; output/cache mapping and unmap
add 0.307 s in the first order and 0.299 s in the second.

No CPU speed parity or first-audio parity was reached. The free-running audio
correlation to CPU int8 was 0.091479 in both orders, consistent with the
earlier FP16 position-major result; listening/intelligibility was not assessed.
These two pairs support a narrower latency gap, not a production switch or a
causal claim that the vendor changed NPU clocks. LiteRT accepts the option
and inference works; there is no direct vendor acknowledgement in logcat.

## Dynamic INT8 G5 with HIGH_PERFORMANCE

The existing `pt_flowlm_fused_dyn8_all_contiguous_half_g5.tflite` AOT graph
was run with the same native `performance_mode = 3` option and position-major
persistent AHWB cache. Its SHA-256 is
`bc8bd5c19495958dfc5b77915dea314c38a9127b839fd357e076d9038cb56acd`.
Both orderings used the fixed long Alba utterance, seed 42, one energy repeat,
and the CPU dynamic INT8 reference. No graph was recompiled for this probe.

| Order | NPU | CPU | NPU / CPU | Frames NPU / CPU | First audio NPU / CPU | LM graph run NPU / CPU | Thermal |
|---|---:|---:|---:|---:|---:|---:|---:|
| NPU then CPU | 51.653 s | 24.917 s | 2.073x | 753 / 759 | 2.836 / 1.113 s | 35.807 / 14.922 s | 0 |
| CPU then NPU | 52.184 s | 24.668 s | 2.115x | 753 / 759 | 2.802 / 1.125 s | 36.041 / 14.795 s | 0 |
| Mean | 51.919 s | 24.793 s | 2.094x | 753 / 759 | 2.819 / 1.119 s | 35.924 / 14.859 s | 0 |

The NPU produced 60.24 s of audio, six frames or 0.48 s short of the CPU's
60.72 s. The LM input/run/read stages averaged 1.290 / 35.924 / 2.490 s on
NPU and 2.323 / 14.859 / 0.125 s on CPU. The native cache
output-map/cache-map/row-copy/unmap components averaged
0.115 / 0.106 / 1.907 / 0.119 s; the persistent input and packed output
buffers reported AHWB type 2. Candidate Mimi decoder transformer and SEANet
averaged 3.416 / 4.571 s, versus 1.793 / 4.504 s on the CPU-LM arm.
Thus cache row patch cost cannot explain the roughly 21.1 s LM graph-run gap.

The paired WAV comparison returned correlation 0.0807276, lag 0,
SNR 0.0284 dB, high-band error 0.0882 dB, reference HNR 0.7958 dB,
and candidate HNR 0.9283 dB in both orders. These are diagnostics, not an
intelligibility assessment. The incomplete output and 2.094x mean latency
reject this combination. HIGH_PERFORMANCE did not change the dynamic INT8
graph's viability; the default-mode position-major INT8 run was also about
2.13x CPU and six frames short. The native opt-in remains off by default.

The direct instrumentation arguments were:

```text
-e class com.pockettts.FlowLmHarnessTest#npuResidentCacheSpeechPair
-e workload long -e voice alba -e seed 42 -e energyRepeats 1
-e order npu-cpu   # repeat with cpu-npu
-e npuResidentCache false -e npuSliceCache true
-e npuPositionMajorCache true -e verifyNpuSliceRows false
-e g5HighPerformance true
-e npuGraph pt_flowlm_fused_dyn8_all_contiguous_half.tflite
```

Reports and WAVs were pulled locally to ignored
`build/int8-highperf/speech-20261006-162459-489/` and
`build/int8-highperf/speech-20261006-163012-588/` in this worktree.
Each directory contains `report.txt`, `npu.wav`, and `cpu.wav`; the reports
include the raw stage, energy, PSS, quality, and output-file records. The
corresponding device directories are under
`/sdcard/Android/data/com.pockettts/files/flowlm-npu-slice-position-major/`.

## Native API and opt-in path

The pinned `litert-2.2.0.aar` exports `LiteRtCreateOpaqueOptions`,
`LiteRtAddOpaqueOptions`, `LiteRtCreateModelFromFile`,
`LiteRtCreateCompiledModel`, and `LiteRtRunCompiledModel`. Its Kotlin
`CompiledModel.Options` has no Google Tensor setting. The upstream
[v2.2.0 Google Tensor options source](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/c/options/litert_google_tensor_options.cc)
serializes the high-performance enum to the TOML string above under the
`google_tensor` identifier. The experimental JNI bridge passes that string
through the exported generic opaque-options API, checks the payload roundtrip,
and compiles with NPU accelerator bit 4. It uses the existing Kotlin model to
allocate buffers, then runs those buffers through the native compiled model.

`PocketTtsConfig.g5HighPerformance` defaults to false and requires
`npuSliceCache`; only the FlowLM model uses the native run path when enabled.
The harness accepts `-e g5HighPerformance true` for both its prompt and speech
probes. The bridge depends on the pinned 2.2.0 C++ `Environment` and
`TensorBuffer` handle layouts, so it must be revalidated before a LiteRT
upgrade. Its second compiled FlowLM model also adds load/memory cost; the
engine's existing `loadMs["lm"]` records only the Kotlin model creation.

Google Maven published LiteRT 2.3.0 on October 6, 2026. Its
`litert-api-2.3.0.aar` still exposes only CPU, GPU, and Qualcomm Kotlin
options; the new version does not offer a direct Kotlin setting for this
experiment. The API AAR is 945,619 bytes versus 934,666 for 2.2.0; the
2.3.0 runtime AAR is 7,640,140 bytes versus 11,761,541 for 2.2.0. The app
remains pinned to 2.2.0 and its matching Google Tensor dispatch shim and
AOT graph.

## Reproduction

Build the debug app after placing the matching v2.2.0
`libLiteRtDispatch_GoogleTensor.so` in
`app/src/main/jniLibs/arm64-v8a/` (it is gitignored; see
`scripts/fetch_google_tensor_dispatch.sh`). Install with
`:app:installDebug :app:installDebugAndroidTest`; do not use
`connectedDebugAndroidTest`, which removes app-owned model files. For each
order run `FlowLmHarnessTest#npuResidentCacheSpeechPair` directly with:

```text
-e workload long -e voice alba -e seed 42 -e energyRepeats 1
-e order npu-cpu   # repeat with cpu-npu
-e npuResidentCache false -e npuSliceCache true
-e npuPositionMajorCache true -e g5HighPerformance true
-e npuGraph pt_flowlm_fused_fp16_contiguous_no_truncation.tflite
```

The G5 graph SHA-256 was
`1e3142667b944172d1216728bec52bfabe3529ac6606394dee2878c0945b8d08`.
The device report directories are
`flowlm-npu-slice-position-major/speech-20261006-160413-958/` and
`flowlm-npu-slice-position-major/speech-20261006-160840-387/`; their `report.txt`
files contain all stage, energy, PSS, quality, and output-file records.
