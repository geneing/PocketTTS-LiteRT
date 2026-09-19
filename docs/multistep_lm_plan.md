# Plan — speeding up the flow-LM on the GPU (`optim/multistep_lm`)

## Problem

The app's frame graph is `pt_flowlm_fused_fp16.tflite`. Per 12.5 Hz frame the host does:

| input | elements | bytes |
|---|---|---|
| `emb[1,1,1024]` + `cos/sin[1,1,1,64]` + `mask[1,16,1,513]` + `noise[1,32]` | 1 024 + 128 + 8 208 + 32 | ~37 KB |
| `pk[1,96,512,64]` + `pv[1,96,512,64]` | 2 × 3 145 728 | **25.2 MB** |
| output `[1,12321]` = eos ∣ latent ∣ new-k ∣ new-v | 12 321 | ~49 KB |

So **99.9 % of the traffic is the packed KV cache**, and it is re-uploaded *in full* every
single step even though only 96×64 floats (one position) changed. On top of that, each
invocation is a host↔GPU round trip, and the step is ~500 tiny kernels for only 78 MMAC
of math (`README.md`, "Graphs and placement").

Measured (Pixel 10, PowerVR DXT-48-1536, this repo): LM on GPU **0.81× real-time** vs
**1.33×** on CPU, so `Placement.default()` pins the LM to CPU on PowerVR. Mali/Adreno keep
the LM on GPU. The user's instrumentation agrees the cost is **memory transfer + sync**,
not arithmetic: `Profile.lmInMs` (host writes) and `Profile.lmRunMs` (invocation + sync)
dominate; `lmReadMs` is already small after fusion.

Goal: make the LM **faster on GPU than the CPU fallback on PowerVR**, without moving audio
quality away from the gold standard.

## Baseline (M0, Pixel 10)

Committed record: `docs/bench/2026-09-19-pixel10-optim-multistep_lm-bdd161a.txt`. LM micro,
ms per frame (`in` = host writes, `run` = invocation submit, `read` = output readback/sync):

| LM placement | in | run | read | total/frame | RTF | audio corr vs gold |
|---|---|---|---|---|---|---|
| CPU (gold) | 1.5 | **19.7** | 0.1 | 21.3 | 0.83× | gold |
| GPU (fp16 compute) | 12.1 | 1.2 | **37.0** | 50.2 | 0.84× | **0.767** |
| GPU32 (fp32 compute) | 12.5 | 1.3 | **52.6** | 66.4 | 0.65× | **1.0000** |

Two findings that revise the premise:

1. **The bottleneck is the readback/sync, not the upload.** `run()` returns after ~1 ms
   because submission is async; the GPU work is hidden inside the reads. Reading back the
   0.05 MB output costs 37 ms — more than uploading the 25.2 MB KV (12 ms). Amortizing
   *invocations* (option A) is therefore the main lever, ahead of shrinking the upload
   (option A's dynamic-KV variant, option B).
2. **fp16 compute on PowerVR fails the quality gate.** `lm:GPU` scores corr 0.767 vs gold;
   `lm:GPU32` (fp32 compute, same fp16 weights) is exact. On this device the LM must run at
   fp32 compute to be quality-clean, ~30 % slower per frame — so the spikes target **GPU32**
   and the fp16 speed number is not a usable target.

Cost to beat: **21.3 ms/frame** (LM on CPU). With fp32 on GPU, amortizing needs
`(12.5 + 52.6)/N + 1.3 < 21.3` ⇒ **N ≥ 4**; N = 8 gives ≈ 9.4 ms/frame (~2.3× the CPU LM),
which is option A's target.

## Success criteria

1. **Quality gate (non-negotiable).** On the same text/voice/seed, the candidate's audio
   vs the CPU/CPU/CPU gold must satisfy: `corr ≥ 0.995`, HNR within **1.0 dB** of gold,
   length within **5 %**, and Whisper re-transcription reproduces the input text. This is
   the same bar the shipped fused graph already clears (`corr 0.997` vs eager).
2. **Performance.** LM `run` + `in` time per generated frame on GPU is at least **2× better
   than the current fused graph on the same device**, and RTF beats the LM-on-CPU fallback
   on PowerVR. Report per-stage medians over ≥3 repeats.
3. **Parity.** Every new graph passes the `build_pockettts.py` tflite-vs-eager check at the
   usual tolerance before it is ever benchmarked on device.
4. **No regressions.** All existing placements/voices still synthesize; the split
   `pt_flowlm_step`/`pt_flow_head` reference graphs stay untouched and working.

## Branch topology

```
optim/baseline
└── optim/multistep_lm      (umbrella: this plan + shared bench/quality harness)
    ├── optim/multistep     (option A: multi-step graphs — amortize sync + transfer)
    └── optim/litertlm      (option B: LiteRT-LM persistent-KV / signatures spike)
```

Spikes branch off the umbrella and merge back only if they pass the quality gate and win on
speed. The umbrella is where the harness and the results index live, so both spikes
benchmark identically.

## Shared harness (lands on `optim/multistep_lm` first)

Do this before either spike so results are comparable.

- **LM micro-benchmark mode.** Add a `Benchmarker` scenario that skips text chunking and
  runs the LM graph for a fixed step count with a fixed seed, reporting per-step
  `lmIn`/`lmRun`/`lmRead` and bytes transferred per step. Today `Profile` only gives
  per-utterance totals; we need per-step and per-invocation numbers to see the amortization.
- **Per-invocation profile fields.** Extend `PocketTtsSynthesizer.Profile` with invocations
  and bytes-in/out so a multi-step graph reports “invocations” not just “steps”.
- **Results store.** `pockettts/docs/bench/` (committed): one text report and the matching
  `bench_*.wav` per run, named `<date>-<device>-<branch>-<shortsha>.txt`. Include the
  placement table, the LM micro-benchmark, and the vs-gold quality block verbatim from
  `Benchmarker`. `benchmark.txt` keeps being written to app filesDir for adb pull; the
  committed copy is the record.
- **Baseline capture.** On Pixel 10, record the current fused graph for GPU32/GPU/CPU
  placements and the gold, so every later number is a delta against a committed baseline.

## Option A — `optim/multistep` (multi-step graphs)

Run **N LM steps inside one invocation**, so one 25.2 MB upload plus one `run()` (and its
sync) covers N frames. Per-step transfer drops to ~`25.2/N` MB; the ~500 dispatches/step
stay, but they no longer each cost a host↔GPU round trip.

### Design

- **Inputs:** `pk/pv[1,96,512,64]` (unchanged), `emb[1,N,1024]`, `noise[1,N,32]`,
  `cos/sin[1,N,1,64]`, `write-mask[1,1,512,1]` (see below), `pos` baked or passed as the
  cos/sin positions.
- **Output:** `eos[1,N]`, `latent[1,N,32]`, and the **new K/V only** —
  `nk/nv[1,96,N,64]` (2 × 49 KB × N/1 at N=32 ≈ 1.5 MB, still tiny). The host splices the
  new positions into its own mirror of the KV; it never reads the 25 MB back.
- **In-graph KV append without banned ops.** The GPU delegate rejects `GATHER`/`SELECT`/
  `WHERE`/`CAST`/`GELU`/rank>4 (`scripts/build_pockettts.py` `BANNED`). Write position
  `off+i` as `kv = kv*(1-m_i) + new_i*m_i` with `m_i` a host-supplied one-hot column
  broadcast over the heads — `MUL/SUB/ADD` only, rank 4. (Alternative: keep the existing
  `concat` trick and fold the written row back with the same mask.)
- **EOS.** Emit the per-step eos logits and let the host stop; because the model is causal,
  overshooting by up to N−1 frames is harmless — the host just discards the latents after
  the EOS step. Cap N so `off + N ≤ PMAX`.
- **Variants to measure** (N = 1 is the current graph, the control):
  `N ∈ {2, 4, 8, 16, 32}`, fp16 weight-cast like the existing graphs.
- **Prefill sub-variant (stretch).** Prompt tokens are ~40–55 sequential steps, each paying
  the full upload. A fixed-`P` graph that consumes `emb[1,P,1024]` and writes P KV rows in
  one shot is the same machinery at a fixed position range; pad short prompts and mask the
  pad positions out of the decode attention. Measure separately.
- **Dynamic-KV sub-variant (stretch, higher risk).** Upload only `[1,96,off,64]` so the
  upload itself shrinks. Needs a dynamic dim, which the app has avoided; test whether the
  GPU delegate accepts it before investing. If it works it is the biggest single win and
  subsumes most of option B's motivation.

### Risks

- Unrolled graph size / compile time grows with N (the fused graph is already 338 MB f32).
  Gate on a per-N binary-size and load-time budget.
- The `write-mask` arithmetic must be fp16-safe and must not perturb the already-written
  prefix (exact 0/1 masks, not softmax-style).
- Some PowerVR kernels may be per-dispatch-bound regardless of N; if per-step `run()` doesn't
  amortize, that is itself the finding to report.

## Option B — `optim/litertlm` (persistent-KV / signature spike)

Research spike, code only if the feasibility gate passes. The opportunity is that
LiteRT-LM's `LlmLiteRtCompiledModelExecutor` already solves the re-upload problem: it keeps
the KV cache in **persistent device buffers** (double-banked, `inplace_update`) and runs the
whole prefill/decode loop in C++, with no host round trip per step, using named
`prefill`/`decode` signatures and `kv_cache_k_*`/`kv_cache_v_*` I/O, plus an
`add_values_to_cache` / `param_tensor` path and dynamic KV.

### Questions to answer (with evidence, no code required to conclude "no")

1. Can the LiteRT-LM executor drive a model that is **not** token-logits based — i.e. one
   whose “token” input is a 1024-d embedding and whose output is a 32-d latent plus EOS?
   (`input_embeddings` exists, but it is still token-indexed.)
2. Where would the **flow head, the Gaussian noise draw and the host-side latent→embedding
   projection** live? They must stay host-side for reproducibility, but the executor owns
   the decode loop — is there a hook, or must the head be baked into the decode graph?
3. Is the persistent-KV machinery usable **without** the full LLM `Pipeline`/`.litertlm`
   bundle (via `litert::Model` signatures + `CreateInputBuffer` + the custom kernels), and
   is any of it reachable from the Android/Kotlin `CompiledModel` API the app uses?

### Decision gate

- **Pass** → prototype the LM as a LiteRT-LM model and benchmark it exactly like option A.
- **Fail** → write up which mechanism is unavailable, and carry the useful half (persistent
  buffers / dynamic KV) into option A's dynamic-KV sub-variant. A negative result is a
  valid spike outcome and gets committed.

## Benchmark protocol (both options, identical)

- Device: Pixel 10 (`frankel`, Tensor G5, PowerVR DXT-48-1536) unless a second device is
  available for cross-checking.
- Fixed text, voice (`alba`) and `noiseSeed`; ≥3 repeats; report medians.
- Placements to compare: `GOLD` (CPU/CPU/CPU), `lm_cpu_dec_gpu`, `shipped_gpu_lm`,
  `all_gpu`, plus `GPU32` for the LM.
- Acceptance requires the quality block **and** the perf numbers in the same report; a fast
  graph that fails the gate is a no-go.

## Checkpoints (commit points)

1. **M0** — plan + harness + baseline capture on `optim/multistep_lm`.
2. **M1** — multi-step export passes CPU/tflite parity in `build_pockettts.py` (no device yet).
3. **M2** — first on-device benchmark of the multi-step graph + quality verdict.
4. **M3** — LiteRT-LM feasibility write-up + gate decision.
5. **M4** — winning variant hardened (N chosen, placement updated), merged to
   `optim/multistep_lm`; results appended to the index.

### M1 result (`optim/multistep`)

`pt_flowlm_ms{4,8}{,_fp16}.tflite` exported with the one-hot write-mask KV append and
new-K/V-only output. GPU-clean (no banned ops, all tensors rank ≤ 4); fp16 size 169.7 /
170.2 MB — essentially the single-step graph's 169.2 MB, because the weights are shared
across the unrolled steps. Parity: the eager N-step unroll is **bit-identical** to the
sequential 1-step loop (latent and eos max|d| `0.0`), and both tflite variants match eager
with corr 1.000000 (max|d| 1.5e-05 / 1.6e-05 at N=4, 2.2e-05 / 1.5e-05 at N=8).

Open before M2: the graph is decode-only (frame 0 from the host, frames 1..N−1 fed back
in-graph), so the text prompt still runs step-by-step; and the host must cap N at
`PMAX − pos`.

### M2 result (`optim/multistep`)

`docs/bench/2026-09-19-pixel10-optim-multistep-d2ac47a.txt`. LM cost per frame (ms,
in/run/read) and full-pipeline RTF on the Pixel 10:

| LM config | in | run | read | total/frame | RTF | corr vs gold |
|---|---|---|---|---|---|---|
| CPU (gold) | 1.6 | 19.2 | 0.1 | 20.9 | 0.84× | gold |
| 1-step GPU | 12.4 | 1.2 | 37.4 | 51.0 | 0.80× | 0.767 |
| ms4 GPU | 3.4 | 0.5 | 28.9 | 32.8 | 0.96× | **0.8334** |
| ms8 GPU | 1.7 | 0.3 | 27.2 | 29.2 | 1.02× | 0.6235 |
| 1-step GPU32 | 12.7 | 1.3 | 52.6 | 66.7 | 0.66× | 1.0000 |

Multi-step does what it was built for: the 25.2 MB upload amortizes (12.4 → 1.7 ms/frame)
and the ~14 ms/invocation sync disappears (51.0 → 29.2 ms/frame, 1.75×). Quality at N=4 is
*better* than the accepted N=1 (0.8334 vs 0.767); N=8 drifts (0.6235) as the in-graph fp16
latent projection compounds.

It does not reach the goal on this device. Fitting read against N gives ≈ 25 ms/frame of GPU
execution (the ~500 dispatches) plus ≈ 14 ms/invocation overhead, and 25 > the CPU LM's 20.9
ms/frame, so GPU+ms still loses to the shipped `lm_cpu_dec_gpu` (1.44×). Two further costs:
the ms graphs load slowly on the GPU delegate (4.2 s at N=4, 14.4 s at N=8 — the serialized
program cache is mandatory once this is default), and the app must hold both the 1-step and
ms graphs.

Conclusion for this spike: keep N=4, but PowerVR stays LM-on-CPU. The multi-step win is a
per-invocation-overhead fix, so it should be re-measured on Mali/Adreno, where the 1-step LM
already beats CPU — and the remaining 25 ms/frame of GPU dispatch is what option B
(persistent KV / no re-upload) would have to attack next.

Commit at every checkpoint (and at any surprising intermediate result); each committed
benchmark report is immutable — new runs add a file rather than editing an old one.

## Risks / fallbacks

- **PowerVR may stay transfer-bound.** If neither option beats the CPU fallback on the
  Pixel 10, the honest deliverable is the measurement plus the finding, and the LM stays
  pinned to CPU there while Mali/Adreno still benefit from the smaller N-step transfer.
- **Dynamic shapes may be rejected** by the GPU delegate; then only the amortization half of
  the win is available (option A without the shrink).
- **Quality gate is the backstop.** Any rewrite that trips the GELU/RoPE/mask numerics
  (fp16-safe masks, exact 0/1 writes) is reverted rather than tuned past the gate.
