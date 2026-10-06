# FlowLM option 4: GPU cache chain gate

Status: **prepared, not passed**. No Pixel 10 timing or quality result is claimed
by this record. The production CPU dynamic-int8 FlowLM stays the control.

## Fixed-shape experiment

`scripts/export_gpu_cache_probe.py` exports `pt_gpu_cache_probe.tflite` without
Pocket TTS weights and checks its positional I/O and row-update math through
the CPU TFLite interpreter. It has a combined K/V cache of `[1,192,512,64]` float32
(25,165,824 bytes), a `[1,192,1,64]` new row, and a one-hot position mask
`[1,1,512,1]`. It returns the full updated cache and one float for host
readback. The gate in `FlowLmHarnessTest.gpuCacheChainGate` alternates two
output buffers as the next input, writing only the row and mask and reading only
the scalar per step. It checks every cache element after the last invocation.

Public API traffic is 49,152 bytes for the row, 2,048 for the mask, and 4 for
the scalar: **51,204 bytes per step**, plus one initial 25,165,824-byte cache
write and one final cache read for correctness. The test records cold model
load, first-step write/run/read, warmed write/run/read, final cache read, and
maximum cache error separately. `run()` alone is not an accepted GPU time.

The [LiteRT GPU documentation](https://developers.google.com/edge/litert/next/gpu)
describes `TensorBuffer::CreateFromGlBuffer`, `GetOpenClBuffer`, and async
execution in C++. The pinned `com.google.ai.edge.litert:litert-api:2.2.0` AAR's
`TensorBuffer` JVM class exposes typed array read/write methods, with no
public GL/OpenCL buffer constructor, handle query, or async method (`javap` on
its `classes.jar`). Its `CompiledModel.run` does accept caller-supplied buffer
lists. Both `litert:2.2.0` and `litert-api:2.2.0` AARs package `.so` files but
no C++ headers. A successful Kotlin chain therefore proves API compatibility
and reduces **public host I/O**; it cannot by itself prove GPU memory residency
or exclude an internal 25 MB device or host copy. That requires the native C++
buffer API and a Pixel GPU trace. The current FlowLM graph also emits only
one-row K/V updates, so its output cannot be fed directly into its full-cache
input. Do not port FlowLM or try dynamic shapes before this gate passes.

## Reproduction when the Pixel 10 and Linux converter are available

```bash
PT_OUT=$(pwd)/scripts/out python scripts/export_gpu_cache_probe.py
adb push scripts/out/pt_gpu_cache_probe.tflite /sdcard/Android/data/com.pockettts/files/
./gradlew :app:installDebug :app:installDebugAndroidTest
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuCacheChainGate \
  -e cacheProbeBackend CPU -e cacheProbeSteps 32 \
  com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuCacheChainGate \
  -e cacheProbeBackend GPU -e cacheProbeSteps 32 \
  com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuCacheChainGate \
  -e cacheProbeBackend GPU32 -e cacheProbeSteps 32 \
  com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb pull /sdcard/Android/data/com.pockettts/files/flowlm-harness/gpu-cache-gate-gpu.txt
```

Run CPU, GPU, and GPU32 separately as above. Record delegate partition count,
rejected ops, kernel count, memory, and whether output buffers remain GPU backed
between calls. If the GPU delegate rejects this graph or a cache buffer cannot
be bound as the next input, stop the option. If it accepts the chain, compare a
native GL/OpenCL-backed path and device trace before claiming saved transfers.
Only after the residency gate passes should the option proceed to FlowLM
integration, N=2/4 unrolls, and the full benchmark and listening protocol in
`flowlm-pixel10-optimization-research.md`.

## Host validation on this branch

| Check | Result |
|---|---|
| Python syntax, `py_compile` | Passed |
| `:app:compileDebugAndroidTestKotlin` | Passed with Gradle 9.6.0 |
| Probe graph export | Blocked: WSL service timed out; no converter environment on native Windows |
| Pixel 10 cache chain, time, energy, quality | Not run; phone allocated to other option tests |

The `gpuResidentCacheBufferApiGate` test in the same harness remains available
to log the JVM API on the device. Its report has no residency measurement.
