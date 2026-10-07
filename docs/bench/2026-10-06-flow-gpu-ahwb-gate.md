# Pixel 10 GPU AHWB cache gate

Status: **stopped at buffer compatibility**. This is an opt-in LiteRT 2.2.0 C
API probe, not a speech benchmark or a performance result.

The position-major fused FP16 FlowLM graph
`pt_flowlm_fused_fp16_contiguous.tflite` was installed on Pixel 10
(`57220DLCR002R6`), with SHA-256
`1d6fe48ed4435b415970a240916a726f1b8994036aacf3c009085722a8a632c7`
matching the local artifact. The native runner compiled the graph with the
public C API and queried `LiteRtEnvironmentSupportsAhwbClInterop` and the
compiled model's per-tensor buffer requirements. Its short prompt gate stopped
before invoking the graph:

```text
ahwbClInterop=false
input0..input6 supported types=14 (OpenClBufferPacked)
output0 supported types=14 (OpenClBufferPacked)
input4/input5 K/V requirement=12,582,912 bytes each, alignment=64
cacheAhwbSupported=false
```

The Pixel OpenCL delegate cannot accept AHWB K/V inputs through this supported
LiteRT path. The probe uses exact v2.2.0 public C declarations resolved from
the AAR's `libLiteRt.so`; the AAR packages that runtime library but no C/C++
headers. It does not use the existing NPU bridge's private Kotlin wrapper
layout. The full report and LiteRT logcat are preserved under ignored
`scripts/out/gpu-flowlm/ahwb-gate/`.

Next candidate, if attempted, must use the delegate's required
`OpenClBufferPacked` type and public OpenCL buffer APIs. Reusing the AHWB
approach without a supported AHWB/CL interop route would only hide a copy.
