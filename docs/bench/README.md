# Benchmark results

Committed records of on-device runs. One file per run; runs are immutable — a new
measurement adds a file, it does not edit an old one.

## Naming

```
<YYYY-MM-DD>-<device>-<branch>-<shortsha>.txt
```

- `device` — e.g. `pixel10` (Pixel 10, Tensor G5, PowerVR DXT-48-1536).
- `branch` — branch the APK was built from, slashes flattened (`optim-multistep`).
- `shortsha` — `git rev-parse --short HEAD` of that build.

## What is in a report

The verbatim `benchmark.txt` written by `Benchmarker` (also available in logcat tag
`PocketTTSBench`):

- `device` / `renderer` / text / voice / repeats / seed header.
- Per placement (`gold_cpu`, `lm_cpu_dec_gpu`, `lm_cpu_all_gpu`, `shipped_gpu_lm`, `all_gpu`):
  load times, per-run audio/frames/ms/RTF with the `lm`/`dec_tx`/`seanet` split, the
  median RTF, the vs-gold quality block, and an **lm micro** line.
- The LM micro line is the one M0 exists for: `N frames / M inv | in X run Y read Z
  ms/frame | A MB in + B MB out per invocation`. For the shipped 1-step graph
  `M == N`; a multi-step graph shows `M == N/step` and the transfer per invocation
  divided by `N/M`.
- Load-time scenarios and GPU program-cache passes.

Matching `bench_*.wav` files live in the app's filesDir (adb-pullable with
`run-as com.pockettts`) and are referenced, not committed, unless a quality
regression needs a permanent audio exhibit.
