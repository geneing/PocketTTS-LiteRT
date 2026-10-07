# FlowLM GPU program-cache startup probe

This is a separate, opt-in **startup** experiment on branch
`codex/flowlm-gpu-program-cache`. It does not change production defaults or
the persistent K/V decode protocol. The probe uses the ordinary fused FP16
graph through LiteRT's Kotlin `CompiledModel` GPU path. The module pins
`com.google.ai.edge.litert:litert:2.2.0`; that version's
[`GpuOptions` source](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/kotlin/src/main/kotlin/com/google/ai/edge/litert/Model.kt)
exposes `serializationDir`, `modelCacheKey`, and `serializeProgramCache`.
The engine already uses these public fields when an opt-in `gpuCache` directory
is configured. This test supplies fresh directories only to the harness.

`FlowLmHarnessTest#gpuProgramCacheColdWarm` ran on Pixel 10 serial
`57220DLCR002R6` with graph `pt_flowlm_fused_fp16.tflite` SHA-256
`1cf47aa0668bb4c6238c869db8902682377727addc0cb8d8f938b63a2a0f2b01`.
For each of two independently new cache directories, it compiled and ran a
three-token Alba prompt, closed the model, then compiled and ran again using
the same directory and model cache key. All four runs delegated **715/715**
nodes to one `LITERT_CL` partition, and all four output tensors were exactly
equal. Each fresh directory produced one
`pt_flowlm_fused_fp16.tflite_mldrift_program_cache.bin` file of
170,353,352 bytes, retained locally in the ignored probe artifact directory.

| Cache pair | Phase | LM initialization | First run + output sync | Later run + output sync, average | First dispatch only |
|---|---|---:|---:|---:|---:|
| 1 | Cold | 1,339.24 ms | 65.95 ms | 34.23 ms | 10.09 ms |
| 1 | Warm | 764.10 ms | 68.08 ms | 39.11 ms | 8.70 ms |
| 2 | Cold | 1,453.20 ms | 40.89 ms | 31.95 ms | 0.54 ms |
| 2 | Warm | 735.20 ms | 119.63 ms | 56.66 ms | 10.15 ms |

Cache-file creation and attempted reuse are confirmed by the file's presence
after the cold pass and the same cache path/key on the warm pass. There is no
public cache-hit counter in this harness, so a hit is inferred from the load
reduction rather than observed directly. Warm initialization was 575/718 ms
faster, 43%/49% respectively. First and later synchronized token timings were
variable and show **no decode-throughput improvement**. These are short
in-process runs, so driver/process caches and scheduling could also affect
the loading difference; this does not establish a cold-process startup result.
It also does not benchmark the separate native OpenCL K/V runner, which does
not currently expose equivalent program-cache options.

The complete report, both generated cache binaries, and filtered delegate
logcat are in ignored `scripts/out/gpu-flowlm/program-cache-probe/` and
`scripts/out/gpu-flowlm/program-cache-logcat.txt`.

Reproduce after installing both debug APKs and retaining external model files:

```text
adb -s 57220DLCR002R6 shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuProgramCacheColdWarm -e gpuGraph pt_flowlm_fused_fp16.tflite com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
```
