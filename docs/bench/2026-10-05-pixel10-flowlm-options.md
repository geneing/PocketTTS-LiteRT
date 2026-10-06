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

## Optimization attempts

Each option has its own branch and bench report. Device tests are run serially because they share this Pixel 10. “Pending” means no performance claim is made.

| Option | Branch | Current implementation / gate | Pixel 10 result |
|---|---|---|---|
| 1. Persistent NPU KV cache | `codex/flowlm-option-1-npu-cache` | Tiny fixed-shape two-bank cache-chain proof | Pending device buffer-chain and transfer measurements |
| 2. CPU int8 prompt prefill buckets | `codex/flowlm-option-2-int8-prefill` | Repair per-row quantized prefill parity; fixed prompt buckets | Pending quantized row parity and device timing |
| 3. KV capacity buckets | `codex/flowlm-option-3-kv-capacity` | Smaller static capacities with 512 fallback and capacity checks | Pending safe-boundary, memory, and device timing |
| 4. GPU resident KV cache | `codex/flowlm-option-4-gpu-cache` | Tiny GPU buffer-chain feasibility gate | Pending device transfer and latency measurements |
| 5. Mobile-oriented architecture | `codex/flowlm-option-5-architecture` | Feasibility/stop assessment; requires trained model changes | Pending agent report; no replacement model may be inferred from shape-only edits |
| 6. CPU int8 retuning | `codex/flowlm-option-6-cpu-retune` | Thread/precision choices with int8 control | Pending paired full-pipeline timing and quality |

No option is labeled a win until correctness, complete speech, and paired device performance pass the acceptance protocol in [`flowlm-pixel10-optimization-research.md`](../flowlm-pixel10-optimization-research.md).
