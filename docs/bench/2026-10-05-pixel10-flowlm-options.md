# FlowLM Pixel 10 optimization results

Date: 2026-10-05  
Device: Pixel 10 (`frankel`), Android 17 / API 37, fingerprint `google/frankel/frankel:17/CP3A.260905.009/16091614:user/release-keys`  
Target: Tensor G5, PowerVR DXT-48-1536  
Control LM: `pt_flowlm_fused_dyn8_all.tflite` on CPU; reference graph: `pt_flowlm_fused_fp16.tflite` on CPU.

## Control measurements

The text probes used `FlowLmHarnessTest.runTextPromptHarness`, the same seedless prompt inputs, voice KV prefix, and graph pair for each run. The harness times `CompiledModel.run()` only: its `runMs` excludes input `writeFloat`, output `readFloat`, and host KV update costs. Treat these as step-compute diagnostics, not end-to-end prefill timings.

| Prompt regime | Voice | Tokens / occupied KV positions | CPU int8 run | CPU fp16 run | int8 vs fp16 output correlation |
|---|---|---:|---:|---:|---:|
| Short (`Hello`) | alba | 1 / 127 of 512 | 13.68 ms (13.68 ms/token) | 24.89 ms | 0.99994582 |
| Medium | mary | 46 / 172 of 512 | 540.15 ms (11.74 ms/token) | 1,023.54 ms | 0.99993606 |
| Near capacity (`hello` repeated) | charles | 360 / 486 of 512 | 4,158.93 ms (11.55 ms/token) | 8,103.37 ms | 0.99986973 |

The near-capacity probe leaves 26 positions after the 126-token Charles voice prefix. It tests prompt-cache writes and does not generate speech from that prompt.

The int8 graph checksum is `895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9`; fp16 reference checksum is `1cf47aa0668bb4c6238c869db8902682377727addc0cb8d8f938b63a2a0f2b01`.

The control text probes passed. All three int8 outputs are highly correlated with CPU fp16; the table does not assert audio quality or parity with eager PyTorch.

## Full-pipeline CPU int8 control

`PowerBenchmarkTest.longParagraphPlaybackPower` completed on the production placement `lm:CPU dectx:NPU dec:GPU`, decoder variant `default`, using the fixed 201-word paragraph (1,168 characters), voice alba, and seed 42. The paired takes produced the same 60.72 s / 759-frame output, with correlation 1.000. The test played the reference and synthesized take while collecting audio-only-subtracted PowerMonitor readings.

| Measure | Result |
|---|---:|
| Inference time | 26.386 s |
| Audio duration | 60.72 s |
| RTF | 2.301x |
| LM / Mimi decoder transformer / SEANet stage time | 15.752 / 2.077 / 4.477 s |
| First-audio latency | Not captured by this control test |
| Incremental CPU/0 energy estimate | 10.861 J |
| Incremental GPU/0 energy estimate | 4.823 J |
| Incremental TPU/1 energy estimate | 1.596 J |

The PowerMonitor domains overlap; do not add these rail values into a single device-energy total. The device was charging at 80%, screen interactive, and thermal status 0 after the run; `soc_therm` was 36.8 C. Energy is an estimate from paired audio-only subtraction. This control is a repeatability check, not an optimized-vs-control A/B.

The played reference and repeat WAVs are available locally at `scripts/out/flowlm-pixel10-control-reference.wav` and `scripts/out/flowlm-pixel10-control-repeat.wav` (ignored build artifacts).

## Option 1: NPU cache-chain first gate

The tiny fixed-shape graph `pt_npu_cache_chain_g5.tflite` (SHA-256 `a5fb30d0c684c29b133f1e5cf23a80b82c318aa37ac058865b79f3b1b8f43282`) compiled as 9/9 ops in one Tensor G5 partition. Its CPU oracle matched the full cache and scalar for three steps. On-device instrumentation then passed 32 steps in two-bank-chain, full-host-write, and same-buffer-alias modes.

| Mode | Host cache bytes/step | Full cache input write | Row write | NPU run | Scalar read/sync |
|---|---:|---:|---:|---:|---:|
| Two-bank chain | 49,152 B | 0 ms | 0.114 ms | 9.137 ms | 0.148 ms |
| Full-host-write control | 25,214,976 B | 11.816 ms | 0.137 ms | 9.416 ms | 0.164 ms |
| Same-buffer-alias probe | 49,152 B | 0 ms | 0.129 ms | 9.610 ms | 0.169 ms |

The graph uses a 25,165,824-byte FP32 cache and 49,152-byte update rows; the one-time initial cache write was 4.933 ms in the chain run. LiteRT reported compatible input/output buffer requirements `[Ahwb, DmaBuf]`, but its Kotlin `TensorBuffer` API did not expose the actual allocated type. The dispatch log confirms one NPU partition but does not reveal whether the runtime copies the full cache inside device memory; it also logged contradictory requirements for the scalar output (a 4-byte tensor versus a reported 64-byte requirement). Device-side copy volume, NPU energy, and full FlowLM timing remain unverified, so this is a successful host-transfer prototype and an incomplete deployment gate, not a speed/energy win.

The three device reports are under `/sdcard/Android/data/com.pockettts/files/flowlm-harness/cache-*/report.txt`; the installed graph and option 1 instrumentation are isolated to that branch's test harness.

## Optimization attempts

Each option has its own branch and bench report. Device tests are run serially because they share this Pixel 10. “Pending” means no performance claim is made.

| Option | Branch | Current implementation / gate | Pixel 10 result |
|---|---|---|---|
| 1. Persistent NPU KV cache | `codex/flowlm-option-1-npu-cache` | Tiny fixed-shape two-bank cache-chain proof | 32-step gate passed; host transfer 49 KB vs 25.2 MB, device-side copies and energy unverified |
| 2. CPU int8 prompt prefill buckets | `codex/flowlm-option-2-int8-prefill` | Quantized P1/P8/P16 headless prefill signatures | Pixel matrix passed exact K/V and first-decode parity for two voices, 1-385 tokens; peak RSS 2,384,056 KiB. Every tested prefill case was slower than sequential int8 (0.531-0.925x); do not enable |
| 3. KV capacity buckets | `codex/flowlm-option-3-kv-capacity` | Smaller static capacities with capacity checks and 512 control | Named-buffer mask/KV regression check passed. Short streaming A/B in both orders on two voices: 256 graph synthesis medians 16-26% faster than 512 with the same frame counts and waveform correlation >0.99999999999998. 46-token Mary case rejected at 256 for its planned frame budget; medium/long and energy checks remain |
| 4. GPU resident KV cache | `codex/flowlm-option-4-gpu-cache` | Export/API feasibility gate | Stopped before artifact/device run: export service timed out; required GPU buffer interop remains unproven |
| 5. Mobile-oriented architecture | `codex/flowlm-option-5-architecture` | Feasibility/stop assessment requiring trained model changes | Stopped: no training corpus, pipeline, checkpoint, or held-out quality suite available; no model or device performance result |
| 6. CPU int8 retuning | `codex/flowlm-option-6-cpu-retune` | Thread counts 2/4/6 plus selective fp32 EOS gate | 18 prompt-only pairs were exact; 6 threads sped 385-token prompt compute 19-23%, but full-pipeline runs were slower in every completed short/medium/long case. Selective EOS export quantized to the baseline graph |
| 7. G5 NPU slice cache | `codex/flowlm-kv-slice` | Full FlowLM graph on NPU; persistent AHWB K/V inputs with native per-position updates | Group-major: two long pairs, mean 1.30x vs CPU. Position-major: two more long pairs, mean 1.31x; host input traffic fell about 104x, row patch fell to 1.78 s. Still slower than CPU |
| 8. G5 dynamic INT8 | `codex/flowlm-g5-int8` | W8/float-activation candidates; group-major `no_truncation` and position-major `half` | Long Alba group-major pairs averaged 52.224/25.021 s NPU/CPU (2.09x); position-major averaged 53.074/25.206 s (2.11x). Both emitted 753/759 frames in both orders. Fails speed and completion; see [candidate report](2026-10-06-flow-g5-int8-candidate.md) |
| 9. G5 static INT8 | `codex/flowlm-g5-static-int8` | Calibrated W8/A8 NPU recipe with float external tensors | Host quality failed at step 0 (latent corr 0.0105, EOS delta 6.465); 32-step minimum corr -0.1744. Stopped before AOT or Pixel test; see [candidate report](2026-10-06-flow-g5-static-int8-candidate.md) |
| 10. G5 `HIGH_PERFORMANCE` runtime mode | `codex/flowlm-g5-high-performance` | Native LiteRT C opaque `google_tensor` option, `performance_mode=3`; harness-only opt-in | Two long Alba pairs averaged 29.827 / 25.052 s NPU/CPU (`1.191x`), all 759 frames; mean first audio 1.524 / 1.137 s. Improves NPU time 8.7% over option 7 position-major FP16, but still misses parity; see [candidate report](2026-10-06-g5-high-performance.md) |
| 11. G5 FP16 `half` truncation | `codex/flowlm-g5-half-truncation` | Existing position-major FP16 graph compiled with AOT `--truncation half` | Two long Alba pairs averaged 32.066 / 24.782 s NPU/CPU (`1.294x`); both emitted only 750/759 frames. Similar to default FP16 NPU speed and fails completion; see [candidate report](2026-10-06-flow-g5-fp16-half-truncation-candidate.md) |
| 12. G5 static W8/A16 | `codex/flowlm-g5-static-w8a16` | Calibrated AEQ W8/A16; all-op, FC-only, and selective FFN variants preserve float32 I/O | Selective FFN12 compiled 694/694 ops in one partition. Two long pairs averaged 32.575 / 24.488 s NPU/CPU (`1.330x`), output 769/759 frames (10 extra); first audio 1.599/1.128 s. Still misses parity; speech completeness/listening unverified. See [initial report](2026-10-06-flow-g5-static-w8a16-candidate.md) and [selective/AOT/Pixel report](2026-10-06-flow-g5-selective-w8a16-candidate.md) |
| 13. G5 dynamic INT8 + `HIGH_PERFORMANCE` | `codex/flowlm-g5-high-performance` | Existing position-major W8/float AOT artifact with native `performance_mode=3` | Two long Alba pairs averaged 51.919 / 24.793 s NPU/CPU (`2.094x`), 753/759 frames. About 2% faster than default-mode INT8, still fails speed and completion; details in [runtime report](2026-10-06-g5-high-performance.md) |

### Option 6: CPU thread tuning

The six-thread full-pipeline harness used the production graph and placement,
seed 42, A/B/B/A order, and two samples per arm. Every completed pair had
audio correlation 1.000. First-audio p50/p95 values are descriptive at n=2.

| Voice / regime | Audio | Default inference | 6-thread inference | First audio default -> 6 threads, p50/p95 | Incremental energy default -> 6 threads |
|---|---:|---:|---:|---:|---:|
| Alba / short | 1.84 s | 817.5 ms | 1,241.5 ms | 818/836 -> 1,242/1,469 ms | 0.00 -> -17.83 J (invalid) |
| Marius / short | 1.52 s | 770 ms | 949 ms | 771/782 -> 950/1,011 ms | 0.00 -> 14.41 J (invalid) |
| Alba / medium | 7.68 s | 3,348 ms | 3,635 ms | 1,571/1,744 -> 1,405/1,424 ms | 0.00 -> 0.00 J (unavailable) |
| Marius / medium | 6.72 s | 2,845.5 ms | 3,326 ms | 1,445/1,500 -> 1,658/1,730 ms | -31.91 -> -30.18 J (invalid) |
| Alba / long | 60.72 s | 25,374.5 ms | 27,866.5 ms | 1,274/1,326 -> 1,267/1,344 ms | 59.29 -> 65.41 J (+10.3%) |
| Marius / long | 55.12 s | 23,244.5 ms | 27,719 ms | 1,190/1,207 -> 1,635/1,710 ms | 56.62 -> 63.28 J (+11.8%) |

Both long runs produced complete output in every arm (Alba: 759 frames / 60.72 s;
Marius: 689 frames / 55.12 s), with PCM correlation 1.000 and thermal status
0 -> 0. PowerMonitor returned zero or negative audio-subtracted energy for
short and medium runs, so those readings cannot rank the candidate. Both long
energy estimates were higher at six threads; the aggregate may include
overlapping rail domains. Device WAVs remain under the app's `power-benchmark`
files directory. Option 3 short-pair WAVs are in
`build/flowlm-worktrees/option3-kv-capacity/scripts/out/option3-speech-device/`;
the final Marius long A/B/B/A WAVs are in
`build/flowlm-worktrees/option6-cpu-retune/scripts/out/option6-long-marius/`.
The full console/logcat record is in ignored `scripts/out`.

## Acceptance coverage

The baseline paragraph test is one long utterance on alba and a same-configuration repeat; it is a control repeatability check, not an A/B against an optimization. Option 2 completed its bucketed prompt/first-decode parity matrix but failed its prefill speed gate, so it was not promoted to full speech testing. Option 3 completed short streaming speech A/B pairs on two voices and both run orders after fixing the named-buffer bug; medium/long speech, energy, and listening remain open. Its 46-token Mary case safely exceeds the 256 capacity with the planned generation budget. Option 1 stopped at the tiny cache-chain gate because actual device-side cache copy volume is unknown. Options 4 and 5 stopped before a Pixel candidate artifact/model was available. Option 6 completed short, medium, and long full-pipeline pairs on both voices; every case was slower at six threads, and long energy estimates were 10-12% higher. Full-pipeline tests for 2/4 threads and listening review remain incomplete.

No option has completed the full acceptance matrix of three prompt lengths, three audio lengths, two voices, reversed paired runs against CPU int8, first-audio percentiles, audio-subtracted PowerMonitor energy, sustained thermal checks, and listening review. No performance win or production-path change is claimed. The exact requirements are in [`flowlm-pixel10-optimization-research.md`](../flowlm-pixel10-optimization-research.md).

## Option 7: Tensor G5 NPU with persistent slice cache

The opt-in prototype runs the fused FP16 FlowLM on the Pixel 10 NPU. It seeds K/V once into persistent LiteRT `AHWB` input buffers, then maps the NPU output and updates only the selected cache position through the native LiteRT 2.2.0 buffer bridge. A preceding short diagnostic verified the K/V rows exactly. The full long run disabled row verification to avoid adding synchronization to its timing. The dispatch report for the model artifact states one G5 partition (715/715 ops).

Both long paired runs used the harness's fixed 201-word paragraph, Alba, seed 42, 759 frames / 60.72 s audio, with the NPU and CPU reference in opposite orders. Both completed without frame truncation. The mean inference times were 32.180 s NPU and 24.808 s CPU int8 (1.30x). These two long pairs are a stable speed result, not the full acceptance matrix: additional prompt/audio regimes, voices, seeds, sustained thermal runs, and listening review remain.

| Pair order | NPU inference | CPU int8 inference | NPU / CPU | NPU LM input / run / read | CPU LM input / run / read | Native row patch |
|---|---:|---:|---:|---:|---:|---:|
| NPU first | 32.906 s | 25.165 s | 1.31x | 1.040 / 16.586 / 2.934 s | 2.407 / 15.309 / 0.129 s | 2.381 s |
| CPU first | 31.453 s | 24.451 s | 1.29x | 0.902 / 16.181 / 2.524 s | 2.418 / 14.984 / 0.110 s | 2.025 s |

In both orders the candidate used 268,080,192 host input bytes versus 27,900,154,944 B for CPU's explicit-cache path (about 104x less). First-audio latency was 1.634 / 1.542 s for NPU and 1.127 / 1.304 s for CPU. Both emitted 759 frames / 60.72 s. The NPU single-G5 partition report was 715/715 ops, and actual buffer types were `[2,2,2]` (AHWB) in both runs.

The native bridge row-copy cost was 2.381 s NPU-first and 2.025 s CPU-first, across 1,107 decode/prompt steps. It writes 96 small, 256-byte rows into group-major cache banks at a 32,768-float stride. LM graph execution remained 1.20-1.28 s slower than CPU; LM output/cache handling remained 2.41-2.81 s slower.

The long free-running waveform correlation was 0.0915 (lag 0) in both pairs. Per the acceptance protocol, this is diagnostic only: listening, intelligibility, completion, speaker stability, and several-seed checks decide speech quality. Short-probe exact row verification checks cache-write integrity, not long-run speech equivalence. PowerMonitor values from each one-repeat pair are recorded in the device reports but are not sufficient to claim an energy result. Thermal status was 0. WAVs and full reports are under ignored `build/flowlm-kv-slice/2026-10-06-{npu-cpu,cpu-npu}/`; device report sources are under `/sdcard/Android/data/com.pockettts/files/flowlm-npu-slice/speech-20261006-082717-444/` and `speech-20261006-084229-847/`.

#### Option 7 position-major cache follow-up

The position-major variant stores each cache as `[1,512,96,64]`, so the bridge
copies the selected position in one contiguous region. It retained the same
full utterance (759 frames / 60.72 s) in both paired orders. The candidate graph
is 172,399,184 bytes, SHA-256
`1e3142667b944172d1216728bec52bfabe3529ac6606394dee2878c0945b8d08`, with
717/717 ops in one G5 partition.

| Pair order | NPU inference | CPU int8 inference | NPU / CPU | NPU first audio | CPU first audio | Native cache patch |
|---|---:|---:|---:|---:|---:|---:|
| NPU first | 33.080 s | 24.827 s | 1.332x | 1.742 s | 1.118 s | 1.778 s |
| CPU first | 32.284 s | 25.157 s | 1.283x | 1.642 s | 1.124 s | 1.783 s |
| Mean | 32.682 s | 24.992 s | 1.308x | 1.692 s | 1.121 s | 1.781 s |

Both runs used the fixed long Alba prompt, seed 42, position-major cache, and
AHWB buffers `[2,2,2]`. Candidate input traffic was 268,080,192 bytes versus
27,900,154,944 bytes for CPU's explicit-cache path (about 104x less). Thermal
status was 0 in both. The cache update fell from 2.03-2.38 s in the group-major
layout to about 1.78 s, but total NPU inference remained 1.31x slower than CPU.
Free-running correlation was 0.09148 and is diagnostic only; no listening or
intelligibility acceptance was completed.

## Option 8: dynamic INT8 G5 conversion

Two dynamic-range W8/float-activation graphs were compiled for Tensor G5 and
run on the same long Alba workload. The group-major graph used `no_truncation`;
the position-major graph used `half` truncation. Both completed in one G5
partition but took about 2.1x as long as CPU int8 and stopped six frames short
of the CPU output. The position-major INT8 graph did not preserve the FP16
position-major speed result. The NPU LM graph-run stage itself took 36.3-37.1 s
versus about 15.0 s on CPU, while host cache input took 0.99-1.21 s on NPU
versus 2.43-2.45 s on CPU. The compiled dispatch is opaque, so the precise
inside-kernel cause is not established; the measured slowdown is in graph
execution, not host cache input. See the [candidate report](2026-10-06-flow-g5-int8-candidate.md)
for stage metrics, host quality checks, exact artifact hashes, and AOT details.
Neither candidate is enabled by the production path.

Current AI Edge Quantizer guidance recommends dynamic W8/float-activation
quantization for CPU/GPU and calibrated static W8/A8 or W8/A16 for NPU
deployment. The dynamic candidate above was still worth measuring because it
preserved the existing float interface, but it is not the recommended NPU
quantization scheme. Option 9 is testing static W8/A8 with float INPUT/OUTPUT
boundaries. It failed the host quality gate before AOT compilation or Pixel
testing. The prior static W8/A16 check is also recorded in the dynamic
candidate report and failed with int16 external tensors and poor short-rollout
agreement. ([AEQ migration
guide](https://developers.google.com/edge/litert/quantization/tflq_to_aeq_migration),
[quantization guidance](https://developers.google.com/edge/litert/quantization/model_optimization))

### Option 9: calibrated static W8/A8

This candidate followed AEQ's NPU-oriented `static_wi8_ai8` recipe, disabling
quantization on INPUT/OUTPUT operations to preserve the existing float32
Android interface. Calibration used Alba's position-major cache, 8 samples
across 96 free-running steps. The 32-step host rollout failed on the very first
step: latent correlation 0.0105, EOS absolute delta 6.465, and K/V max delta
4.769, versus CPU dynamic INT8 latent correlation 0.9998 on that step. It was
not AOT compiled and did not reach the Pixel test gate. Full details and the
opt-in quantization script are on branch `codex/flowlm-g5-static-int8`; the
[host report](2026-10-06-flow-g5-static-int8-candidate.md) records the checksums
and 32-step drift.

### Option 10: Tensor G5 high-performance runtime mode

The pinned LiteRT Kotlin `CompiledModel.Options` does not expose Google Tensor
performance settings, including in the newer 2.3.0 API AAR. An opt-in native
LiteRT C bridge passes the opaque `google_tensor` payload
`performance_mode = 3` to create a second compiled model; the AOT graph is
already targeted at Tensor G5. The one-token Pixel smoke passed with NPU/CPU
latent correlation 0.999989. Two long Alba runs in opposite orders both
emitted all 759 frames. NPU averaged 29.827 s versus CPU int8 25.052 s (1.191x);
first audio averaged 1.524 s versus 1.137 s. The reversed order confirms the
result, but CPU parity is still not reached. Code and harness changes are
isolated on `codex/flowlm-g5-high-performance`; exact run data are in the
[high-performance candidate report](2026-10-06-g5-high-performance.md).
Applying the same setting to the position-major dynamic INT8 graph averaged
51.919 s NPU versus 24.793 s CPU (2.094x), with the NPU again stopping at 753
of 759 frames; this improved the default INT8 NPU timing by only about 2%.

### Option 11: FP16 AOT half truncation

The existing position-major FP16 source was compiled with `--truncation half`.
The compiler completed in 33.3 s with 717/717 ops in one G5 partition. The
172,399,184-byte artifact has checksum
`fb66bc91257207b809a4aab4a77e33168cd105cd4ee266405798a4937ccfdfac`. Two
long Alba pairs in opposite orders averaged 32.066 s NPU versus 24.782 s CPU
(1.294x); both NPU arms emitted 750 of 759 frames. First audio averaged 1.657
versus 1.136 s. The row patch was 1.83-1.84 s, thermal status was 0, and the
free-running waveform correlation was 0.2216 (diagnostic only). It did not
improve materially on default FP16 NPU timing and failed the output-completion
gate. See the [candidate report](2026-10-06-flow-g5-fp16-half-truncation-candidate.md)
for the paired timings, stage metrics, report paths and hashes.

### Option 12: calibrated static W8/A16

AEQ 0.8.0 all-op W8/A16 failed on step zero (latent correlation -0.0736,
EOS delta 6.485); FC-only began at 0.9497 but drifted to a 32-step minimum of
0.0106. A selective variant quantized only 12 of 47 fully connected matrices
(59.6% of dense weights), leaving Q/K/V, attention, EOS, and flow head in
FP32. It reached first-step correlation 0.9998 and a 32-step minimum of
0.6699, versus -0.1247 for the CPU dynamic INT8 host rollout on the same seed.
The G5 compiler placed all 694/694 ops in one partition. Two long Alba Pixel
pairs then averaged 32.575 s NPU versus 24.488 s CPU (1.330x), with first audio
at 1.599 versus 1.128 s. FFN12 emitted ten extra frames (769 versus 759), and
its long free-running waveform correlation was 0.0724; listening and spoken
text completeness remain unverified. The single-repeat energy subtraction was
noisy and does not support an energy claim. It still misses speed parity.
Initial all-op and FC-only details are in the [W8/A16 report](2026-10-06-flow-g5-static-w8a16-candidate.md);
selective coverage, drift, AOT, and Pixel measurements are in the [FFN report](2026-10-06-flow-g5-selective-w8a16-candidate.md).

### How other runtimes manage KV state

- **LiteRT-LM** stores state in persistent `TensorBuffer`s. Its generic runtime supports in-place updates when the backend permits them, ping-pong buffers otherwise, and a GPU-specific policy that avoids passing the cache as an input. Its NPU executor also has a hardware cache-update path (`HWKVCacheUpdate`) separate from the model-side update path; this is the closest reference for our split model/host cache-commit loop. We still need to establish whether that NPU implementation can be reused with the Tensor G5 dispatch. ([state allocation policies](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/executor/litert/state.h), [NPU settings](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/executor/llm_executor_settings.h), [NPU update implementation](https://chromium.googlesource.com/external/github.com/google-ai-edge/LiteRT-LM/+/60de4ed7d2d229001eafdc0714b73d9a26c70ed4/runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.cc))
- **ExecuTorch** recommends cache tensors as model-owned buffers updated by an explicit position input, keeping cache tensors out of the runner's input/output protocol. It provides export-friendly cache-update and attention operators. That is the cleanest model boundary, but it depends on the target backend supporting exported mutable state. ([custom LLM export guide](https://github.com/pytorch/executorch/blob/main/docs/source/llm/export-custom-llm.md), [KVCache module](https://github.com/pytorch/executorch/blob/main/extension/llm/modules/kv_cache.py))
- **OpenVINO GPU** represents state with `ReadValue` / `Assign`, then fuses cache concatenation and state assignment into a `KVCache` operation to avoid update overhead. **OpenVINO Intel NPU** documents the opposite constraint: its driver/runtime does not support state variables, so the compiler lowers them to additional inputs/outputs and copies through an intermediate state buffer. A stateful graph is therefore not by itself proof of zero-copy execution. ([GPU KV cache fusion](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/dynamic_shape/kv_cache.md), [Intel NPU stateful models](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md#stateful-models))
- **LiteRT Kotlin `CompiledModel`** exposes preallocated input/output buffers and runs each invocation against those buffers. The pinned 2.2.0 Kotlin surface has no public output-to-input binding method; our current prototype instead maps and patches the actual AHWB buffers through a version-pinned native bridge. ([NPU buffer flow](https://developers.google.com/edge/litert/next/npu), [pinned Kotlin API](https://raw.githubusercontent.com/google-ai-edge/LiteRT/v2.2.0/litert/kotlin/src/main/kotlin/com/google/ai/edge/litert/Model.kt))

For this model, the next useful runtime experiment is the LiteRT-LM style separation between a one-step inference graph and a hardware/runtime cache commit. The position-major layout already made the cache commit contiguous and reduced it to about 1.78 s per long utterance; end-to-end NPU time remains 1.31x slower than CPU.
