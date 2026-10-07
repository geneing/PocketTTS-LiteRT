# Pixel 10 FlowLM GPU optimization results

**Status: GPU remains slower than CPU dyn8 for long speech.** Persistent OpenCL K/V buffers substantially reduce cache traffic, but the output synchronization/read path dominates. No production placement or default changed.

## Device and comparison protocol

Tests ran on Pixel 10 (`57220DLCR002R6`), Android 17 build `CP3A.260905.009`, using `FlowLmHarnessTest` and the long Alba, seed-42 workload. Long speech pairs used the same CPU decoder-transformer and GPU SEANet placement, with GPU→CPU and CPU→GPU orders where shown. Each arm included warmup, a same-workload baseline, an audio-only power interval, and a measured take. Instrumentation was driven directly so pushed model files remained installed.

The CPU dyn8 graph hash was `895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9`. The standard GPU FP16 graph hash was `1cf47aa0668bb4c6238c869db8902682377727addc0cb8d8f938b63a2a0f2b01`; the position-major FP16 graph used by the OpenCL cache experiments was `1d6fe48ed4435b415970a240916a726f1b8994036aacf3c009085722a8a632c7`.

## Long speech results

| Optimization attempt | GPU measured synthesis | CPU dyn8 measured synthesis | GPU/CPU | Frames (GPU/CPU) | Result |
|---|---:|---:|---:|---:|---|
| Standard GPU, host K/V, GPU→CPU only | 112.693 s | 35.126 s | 3.21× | 754/759 | Baseline; one completed order |
| Persistent OpenCL packed K/V, output lock, GPU→CPU / CPU→GPU | 87.072 / 87.055 s | 34.502 / 34.523 s | 2.52× / 2.52× | 754/759 | Host K/V input fell from about 27.9 GB to 267.9 MB per take; output lock still took 48.4–48.7 s |
| Persistent K/V plus direct 33-float control read, GPU→CPU / CPU→GPU | 86.002 / 86.558 s | 35.105 / 35.009 s | 2.45× / 2.47× | 754/759 | Saved only 0.50–1.07 s versus output lock; no useful placement win |
| Kotlin `GpuOptions.Priority.HIGH`, GPU→CPU / CPU→GPU | 78.906 / 79.192 s | 35.083 / 34.880 s | 2.25× / 2.27× | 753/759 | Same-build default GPU took 79.748 / 79.524 s; HIGH averaged 0.74% faster, within run variation |

The HIGH-priority test used the standard host-K/V path, not the persistent OpenCL cache. Its GPU output WAV was byte-identical across HIGH/default settings and order, but both settings ended six frames before the CPU reference. GPU-versus-CPU waveform correlation is a diagnostic between different FP16/dyn8 free runs, not a listening or intelligibility result.

## Startup, profiling, and compatibility probes

| Probe | Measurement or result | Interpretation |
|---|---|---|
| GPU serialized program cache | Cold init: 1,339 / 1,453 ms; warm reuse: 764 / 735 ms. One 170,353,352-byte cache file per directory. | About 43–49% lower initialization in these short in-process probes. First-token times varied; no steady decode-throughput improvement was established. |
| LiteRT operator profiler | 717/717 nodes delegated to one `LITERT_CL` partition. `UploadOrBindTensorBuffer`: 14.723 ms, 30.8% of summed node time. | Useful upload/bind diagnostic; the single partition did not expose the app's full per-token output-wait cost. Shell timing is separate from foreground app timing. |
| AHardwareBuffer K/V binding | `ahwbClInterop=false`; input requirements were OpenCL packed type 14. | Unsupported by the Pixel's LiteRT GPU interop path. The persistent K/V candidate uses the required packed OpenCL buffers instead. |
| Native C GPU HIGH priority | The pinned AAR did not export the public `Lrt*GpuOptions` builder/setter symbols. | Stopped at the API availability gate; no native-cache HIGH run or performance result. Kotlin HIGH was tested separately. |
| External-weight FD / static memory pool | The public LiteRT 2.2.0 Kotlin options API does not expose `ExternalWeightScopedFileDescriptor` or `MemoryPlanningStrategy.PREALLOCATED_STATIC_POOL`. | No experiment was possible through the documented API. The native cache runner allocates and reuses its input/output buffers; `CompiledModel` loads the model by file path. |

The [LiteRT GPU options API](https://developers.google.com/edge/api/litert/kotlin/com/google/ai/edge/litert/CompiledModel.GpuOptions) documents program-cache and priority controls. The [LiteRT benchmark guide](https://developers.google.com/edge/litert/next/benchmark) documents `--use_profiler=true` and notes that shell-process scheduling can differ from a foreground app; therefore shell profiler numbers are not used as the long-speech performance result.

## Checkpoints and detailed reports

Each option remains independently checkpointed. Branch tips and reports:

| Branch | Tip | Detailed report |
|---|---|---|
| `codex/flowlm-gpu-profile` | `b6f9974` | [Initial GPU baseline](2026-10-06-flow-gpu-profile.md) |
| `codex/flowlm-gpu-kv-optimization` | `3d3dbbc` | [AHWB gate](2026-10-06-flow-gpu-ahwb-gate.md) |
| `codex/flowlm-gpu-opencl-kv` | `721ddd7` | [Persistent OpenCL K/V](2026-10-06-flow-gpu-opencl-kv.md) |
| `codex/flowlm-gpu-opencl-control-read` | `fe90b9d` | [Direct control read and profiler](2026-10-06-flow-gpu-opencl-direct-read.md) |
| `codex/flowlm-gpu-program-cache` | `6b9b7c4` | [Program-cache probe](2026-10-06-flow-gpu-program-cache.md) |
| `codex/flowlm-gpu-priority-high` | `e3dcf3a` | [Native priority API gate](2026-10-06-flow-gpu-native-priority-gate.md) |
| `codex/flowlm-gpu-kotlin-priority-high` | `13054a3` | [Kotlin HIGH-priority comparison](2026-10-06-flow-gpu-kotlin-priority-high.md) |

Reports, raw instrumentation outputs, and WAVs are retained locally under ignored `scripts/out/gpu-flowlm/` paths in the experiment worktree. No GPU candidate was selected as the default because the long speech measurements did not establish speed parity with CPU dyn8.
