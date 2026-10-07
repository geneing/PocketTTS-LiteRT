# Pixel 10 Kotlin GPU HIGH priority FlowLM probe

This is an opt-in trial of LiteRT 2.2.0 Kotlin
`CompiledModel.GpuOptions(priority = Priority.HIGH)` for the standard fused
FP16 FlowLM GPU path. It does not use the native OpenCL persistent K/V cache.
No production placement or graph default changes.

The bundled Kotlin API accepts this priority option. In contrast, the
version-pinned Android AAR omits the public C `Lrt*GpuOptions` symbols needed
to set HIGH on the native OpenCL-cache runner; that separate native gate and
its stop reason are recorded on branch `codex/flowlm-gpu-priority-high`.

## Device, graph, and protocol

- Pixel 10 (`frankel`), serial `57220DLCR002R6`, Android 17,
  `google/frankel/frankel:17/CP3A.260905.009/16091614:user/release-keys`.
- GPU graph: `pt_flowlm_fused_fp16.tflite`, SHA-256
  `1cf47aa0668bb4c6238c869db8902682377727addc0cb8d8f938b63a2a0f2b01`.
  CPU reference: `pt_flowlm_fused_dyn8_all.tflite`, SHA-256
  `895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9`.
- `FlowLmHarnessTest#gpuSpeechPair`, Alba, seed 42, `workload=long`, one
  measured synthesis per arm, short warmup, same-workload baseline and
  duration-scaled audio-only power subtraction. LM is the only changed
  placement/option: both arms put decoder transformer on CPU and SEANet on GPU.
- Same installed `assembleDebug`/`assembleDebugAndroidTest` APKs for HIGH and
  default control. `gpuLmPriorityHigh=true` applies only to the GPU LM;
  omitted flag retains default GPU options. The program-cache directory is
  the same in both modes. Android instrumentation was driven directly so the
  installed model files were retained.
- Filtered device logcat shows **715/715** FlowLM nodes delegated to one
  `LITERT_CL` partition in both modes. The harness reports
  `lm=GPU, dectx=CPU, dec=GPU, dec_w=GPU` for each GPU arm.

## Tensor gate

The opt-in `gpuKotlinPriorityPromptGate` ran three Alba prompt tokens. HIGH
and default GPU outputs matched exactly (correlation 1.0, MAD 0), while HIGH
versus CPU FP16 had correlation 0.99999772 and MAD 0.0013669. The GPU output
buffers each held 147,852 float bytes. On this very short gate, default
input/`run`/read was 30.195/16.652/134.362 ms and HIGH was
28.394/2.111/152.886 ms; complete stage totals were effectively tied.
The raw report/tensors are ignored under
`scripts/out/gpu-flowlm/kotlin-priority-gate/`.

## Long paired runs

The table reports the measured speech takes, not model load or warmup. All
times are seconds. `run()` alone understates GPU time; read/sync and full K/V
input transfer are included in end-to-end synthesis.

| Priority / order | GPU frames | GPU total | GPU first audio | GPU LM input / run / read | CPU frames | CPU total |
|---|---:|---:|---:|---:|---:|---:|
| HIGH, GPU→CPU | 753 | 78.906 | 3.861 | 10.588 / 26.270 / 20.966 | 759 | 35.083 |
| HIGH, CPU→GPU | 753 | 79.192 | 3.980 | 10.794 / 26.739 / 20.555 | 759 | 34.880 |
| Default, GPU→CPU | 753 | 79.748 | 4.088 | 10.789 / 25.876 / 21.857 | 759 | 35.831 |
| Default, CPU→GPU | 753 | 79.524 | 4.044 | 10.837 / 26.782 / 20.890 | 759 | 34.946 |

Every run used 1,101 GPU LM steps and 1,107 CPU LM steps. GPU host input
was 27,748,934,592 bytes and GPU output 54,261,684 bytes in each order,
because this standard Kotlin path still moves full K/V banks per token.
Waveform correlation against CPU dyn8 was 0.077522 (lag 0), SNR 0.026 dB,
high-band error 0.072 dB, and HNR CPU/GPU 0.800/0.656 dB. Both orders
reported thermal status 0→0. The six-frame GPU shortfall is a completion
difference from CPU, so HIGH has no exact-quality parity claim. All four GPU
WAVs were byte-identical (SHA-256
`b6937909440052ac18c31139a67eccaffcbbfe47bf769043b8e4e2048e8c0b53`),
as were the four CPU WAVs (SHA-256
`fef233fc5f2ab7fb28f2fc06cb809d390f388366a7349040470b763581a42343`).

PowerMonitor's duration-scaled audio-only subtraction put GPU/0 incremental
energy at **17.057, 17.811 J** with HIGH and **16.928, 17.003 J** at default
priority, compared with the CPU-reference arms at **4.554, 4.537, 4.683,
4.696 J**. These overlapping rail domains must not be summed; the differences
between priorities are too small and inconsistent for an energy benefit claim.

## Same-build default-priority control

The default controls used exactly the same installed app/test APKs, graph,
program-cache directory, seed, voice, and workload as HIGH. They produced the
same GPU/CPU frames and byte-identical WAVs. HIGH took 78.906/79.192 s across
the two orders; default took 79.748/79.524 s. HIGH averaged **79.049 s** and
default **79.636 s**, a 0.587 s (0.74%) difference. The paired GPU-minus-CPU
gap averaged 44.068 s under HIGH and 44.248 s at default, a 0.180 s
difference. The CPU arms themselves varied by up to 0.951 s across these
sessions. Neither end-to-end timing nor energy establishes a meaningful
priority win. Both priority modes are about **2.25×** slower than CPU dyn8,
and both stop six frames before the reference. Retain HIGH only as an opt-in
diagnostic; do not enable it as a production default.

## Reproduction and artifacts

```text
adb -s 57220DLCR002R6 shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuKotlinPriorityPromptGate com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb -s 57220DLCR002R6 shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#gpuSpeechPair -e gpuLmPriorityHigh true -e gpuGraph pt_flowlm_fused_fp16.tflite -e workload long -e order gpu-cpu -e seed 42 -e voice alba com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
```

Change `order` to `cpu-gpu` for the reverse. Omit `-e gpuLmPriorityHigh true`
for the default-priority control. Device reports and WAVs were pulled to
ignored `scripts/out/gpu-flowlm/kotlin-priority-long-{gpu-cpu,cpu-gpu}/` and
`scripts/out/gpu-flowlm/kotlin-default-long-{gpu-cpu,cpu-gpu}/`. Filtered
backend logcat is in ignored
`scripts/out/gpu-flowlm/kotlin-priority-four-run-logcat.txt`.

The older ordinary GPU run on a different branch/session took 112.693 s;
that is contextual history, not a controlled estimate of the priority effect.
