# Native OpenCL cache GPU HIGH-priority gate

Status: **stopped before compilation** on the pinned LiteRT 2.2.0 Android AAR.
This separate opt-in branch `codex/flowlm-gpu-priority-high` attempted to apply
the public C equivalent of Kotlin `GpuOptions.Priority.HIGH` to the native
persistent-K/V runner, leaving its graph, buffers, output lock, and cache-row
copy protocol unchanged.

The [v2.2.0 public GPU options header](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/c/options/litert_gpu_options.h)
declares `LrtCreateGpuOptions`, `LrtSetGpuOptionsGpuPriority`, and
`LrtGetOpaqueGpuOptionsData`; the public enum value is
`kLiteRtGpuPriorityHigh = 3`. The actual packaged arm64 `libLiteRt.so`
exports `LiteRtCreateOpaqueOptions` and `LiteRtAddOpaqueOptions`, but not the
`Lrt*GpuOptions` builder/setter functions. `FlowLmHarnessTest#gpuOpenClPromptGate
-e gpuPriorityHigh true` therefore stopped immediately with:

```text
GPU HIGH priority unavailable: pinned LiteRT 2.2.0 AAR omits public Lrt*GpuOptions symbols
```

The default-priority gate still passed after making those symbols optional in
the native bridge, proving this branch does not break the existing opt-in
cache path. No HIGH-priority native graph was compiled, no long priority run
was executed, and no priority performance claim is made. The instrumented gate
transcript and extracted AAR export evidence are retained locally in ignored
`scripts/out/gpu-flowlm/native-priority-high-gate.txt` and
`scripts/out/gpu-flowlm/api-lib/`. Passing a hand-constructed opaque GPU
payload would require an unsupported private ABI assumption, so this branch
does not do that.

The pinned Kotlin API does expose `GpuOptions(priority = Priority.HIGH)`;
testing that on the standard Kotlin GPU FlowLM path is a distinct candidate
and does not imply priority was applied to this native cache runner.

Other exposed GPU options were not combined with this trial:
`allowSrcQuantizedFcConvOps` introduces activation quantization (and requires
constant-tensor sharing), which would change the FP16 parity/quality contract.
`preferTextureWeights` chooses a different weight storage layout; its benefit
is hardware dependent and it does not directly remove the measured
output-synchronization bottleneck. Neither was enabled or measured.
