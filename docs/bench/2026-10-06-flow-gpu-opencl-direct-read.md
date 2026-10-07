# FlowLM GPU control-only OpenCL read and op profiler

Branch `codex/flowlm-gpu-opencl-control-read`, implementation checkpoint
`c197a57`, extends the persistent K/V path reported in
`2026-10-06-flow-gpu-opencl-kv.md`. Status: **small synchronization reduction,
not an end-to-end win**. The stable lock-based branch is retained separately.

## Change and parity gate

The native step now issues a blocking `clEnqueueReadBuffer` for only the 33
control floats on LiteRT's borrowed in-order OpenCL command queue. This follows
the delegate's enqueued kernels and precedes the two same-queue K/V row copies;
`clFinish` still completes copies before the next invocation. It avoids
LiteRT's whole-output lock/map/unmap. The model, placements, seed, and
host-side cache protocol are unchanged. All K/V and output buffers remain
type 14 `OpenClBufferPacked` with runtime-checked shapes and sizes.

`FlowLmHarnessTest#gpuOpenClPromptGate` passed on Pixel 10
`57220DLCR002R6`: three-token full tensor GPU-vs-CPU FP16 correlation
`0.99999772`, MAD `0.001367`. Re-seeding and replaying the three tokens with
33-float-only reads gave maximum control difference `0.0` versus the full
GPU read. Direct full-output read time was 148.304 ms total for the three
tokens; control-only read was 121.494 ms. The gate tensors and report are in
ignored `scripts/out/gpu-flowlm/opencl-direct-control-gate/`.

## Long Alba seed-42 paired measurements

Both orders used the same `FlowLmHarnessTest#gpuSpeechPair` warmup, long
repair-fair baseline, audio-only playback subtraction, and measured take. GPU
graph SHA-256 is
`1d6fe48ed4435b415970a240916a726f1b8994036aacf3c009085722a8a632c7`;
CPU dyn8 reference SHA-256 is
`895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9`.
Both arms used CPU decoder transformer and GPU SEANet. Times are seconds.

| Order | Arm | Frames | Synthesis | First audio | LM input | LM run | LM read | GPU control read |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPU→CPU | GPU FP16 direct read | 754 | 86.002 | 4.245 | 5.790 | 2.152 | 51.300 | 47.744 |
| GPU→CPU | CPU dyn8 | 759 | 35.105 | 1.526 | 2.260 | 14.250 | 0.121 | — |
| CPU→GPU | GPU FP16 direct read | 754 | 86.558 | 4.450 | 5.741 | 2.166 | 51.911 | 48.373 |
| CPU→GPU | CPU dyn8 | 759 | 35.009 | 1.506 | 2.290 | 14.071 | 0.123 | — |

The app's installed test APK still labels the last stage `outputLock` in its
raw report; it measures the **direct OpenCL read** in this branch. A subsequent
source-only label edit to `outputSyncRead` did not change the installed native
runner for these two takes. GPU cache-copy enqueue was 0.121/0.114 s and its
wait was 3.434/3.422 s. Host input and output bytes remained 267,892,352 and
145,464 per GPU arm. Both orders ended at 754 GPU versus 759 CPU frames,
with waveform correlation `0.1601784`, lag 0, SNR `0.113 dB`, and thermal
status 0→0. Duration-adjusted, audio-only-subtracted GPU/0 rail energy was
21.602/22.124 J on GPU versus 4.730/4.615 J on CPU; overlapping rail domains
must not be added.

The stable lock-based GPU path measured 87.072/87.055 s in the two orders and
48.425/48.661 s output lock. Direct read measured 86.002/86.558 s and
47.744/48.373 s control read. That is a marginal 0.50–1.07 s synthesis
reduction, while GPU remains **2.45–2.47× slower** than CPU and still has the
same five-frame completion gap. It is not a useful placement win.

Full app reports and WAVs are retained locally, ignored by Git, at
`scripts/out/gpu-flowlm/opencl-direct-long-gpu-cpu/` and
`scripts/out/gpu-flowlm/opencl-direct-long-cpu-gpu/`.

## Official benchmark-op profiler diagnostic

I ran the official [LiteRT `benchmark_model` profiler](https://developers.google.com/edge/litert/next/benchmark)
on the same installed FP16 position-major graph with `--use_gpu=true
--use_profiler=true --num_runs=1 --warmup_runs=1`. The downloadable arm64
binary was the then-current **nightly**, SHA-256
`6126fba881597ae9ad7f713e752813f41349cd4aa68c9f23e5a7aaea54fb59cf`;
it is not the app's pinned 2.2.0 runtime. The shell process does not execute
the app's autoregressive K/V cache protocol, and its scheduling differs from a
foreground app, so its times are diagnostic only.

The benchmark delegated all `717/717` graph nodes to one `LITERT_CL` partition
and reported 393 executed delegate nodes. Its second run reported 153.08 ms
average inference (seven timed runs) and 667.91 ms first warmup. The per-node
profile attributed 14.723 ms (30.8% of summed node time) to
`UploadOrBindTensorBuffer`, 1.776 ms (3.7%) to GPU-to-output download,
and 0.428 ms average across eight transpose nodes (7.2% combined).
Fully connected nodes and their `add` fusions accounted for 28.0% combined.
The profiler therefore confirms nontrivial input binding/upload and transpose
work, but cannot explain the app's whole 47–49 s output wait stage by itself.
The full raw profiler output is retained in ignored
`scripts/out/gpu-flowlm/benchmark-model-gpu-profiler.txt`.

No production defaults changed. The direct-read implementation remains on its
separate branch for reproducibility; the paired result does not justify
selecting GPU for the shipped placement.
