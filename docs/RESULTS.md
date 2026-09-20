# Pocket TTS on-device results — Pixel 10 (Tensor G5)

A one-day sweep of the flow-LM and Mimi decoder across every accelerator and step
count the app can reach: **CPU** (XNNPACK), **GPU** (OpenCL delegate, fp16 weights),
**GPU32** (fp32 compute) and the **Tensor G5 NPU** (EdgeTPU, AOT-compiled by
`scripts/aot_tensor_g5.py`). Timing only; the raw reports carry the quality blocks.

- **Device** — Pixel 10 (`frankel`), Google Tensor G5, GPU `PowerVR D-Series
  DXT-48-1536`, Android 17.
- **Workload** — 95-char text, voice `alba`, seed `20260919`, 28 prompt tokens +
  70 generated frames = 5.60 s audio at 24 kHz.
- **Protocol** — one process per run, gold (CPU/CPU/CPU) first as the audio
  reference, then the candidates; median of 2–3 repeats. See `bench/README.md`.

## How to read the per-step table

The `lm micro` line of a report times each of the three stages the host controls:

| column | what it measures |
|---|---|
| `copy-in` (`in`) | writing host buffers into the graph's input tensors. For the flow-LM this is dominated by the 25.2 MB packed KV. |
| `compute` (`run`) | the `run()` call. **On GPU this only enqueues**; on CPU and NPU it blocks through the compute. |
| `readback+sync` (`read`) | reading the output tensors. It blocks until the device has finished, so **on GPU this carries both the real compute and the sync**; on CPU/NPU it is a pure copyback. |

So the "compute" and "readback" columns are not the same quantity on every row:
GPU compute is `readback+sync` (37.1 / 27.7 / 27.7 ms), and its 1.2 ms `compute`
column is enqueue overhead. CPU and NPU are the mirror image. Repeat-to-repeat
spread is about 5%.

## A. LM per-step cost

128 LM frames, single session (M5), so all accelerators are directly comparable.

| LM graph | accel | steps/invocation | invocations | MB in/inv | copy-in ms/step | compute ms/step | readback+sync ms/step | **total ms/step** | ms/invocation |
|---|---|---|---|---|---|---|---|---|---|
| `fused_fp16` | **CPU** | 1 | 128 | 25.2 | 1.4 | 19.0 | 0.15 | **20.5** | 20.5 |
| `fused_fp16` | **GPU** | 1 | 128 | 25.2 | 12.5 | 1.2 | 37.1 | **50.8** | 50.8 |
| `ms4_fp16` | **GPU** | 4 | 32 | 6.3 | 2.7 | 0.44 | 27.7 | **30.9** | 123.4 |
| `ms8_fp16` | **GPU** | 8 | 16 | 3.2 | 1.5 | 0.28 | 27.7 | **29.5** | 235.7 |
| `fused` (fp32) | **GPU32** | 1 | 128 | 25.2 | 12.9 | 1.2 | 52.7 | **66.9** | 66.9 |
| `fused_fp16_g5` | **NPU** | 1 | 128 | 25.2 | 6.7 | 17.2 | 0.50 | **24.4** | 24.4 |
| `ms4_fp16_g5` | **NPU** | 4 | 32 | 6.3 | 2.1 | 26.9 | 0.55 | **29.5** | 118.0 |
| `ms8_fp16_g5` | **NPU** | 8 | 16 | 3.2 | 1.1 | 26.8 | 0.26 | **28.2** | 225.4 |

What the breakdown says:

1. **The GPU 1-step graph is transfer- and sync-bound, not compute-bound.** 12.5 ms
   copy-in + 37.1 ms wait against 1.2 ms of enqueue: ~97% of the step is spent
   moving or waiting on data. This is why the flow-LM loses to XNNPACK on PowerVR.
2. **Batching amortizes the copy but not the wait.** N=4 and N=8 cut copy-in by
   4.6x and 8.3x, yet `readback+sync` stays at ~27.7 ms/step, so per-step cost
   flattens at ~29.5 ms. That wall is why GPU gains stop at N=8.
3. **The NPU is the opposite shape: compute-bound with a cheaper copy.** Copy-in is
   roughly half the GPU's (6.7 vs 12.5 ms) and there is no readback wall (0.5 ms),
   but its compute is 17.2 ms against the GPU's effective 37.1 ms — and 17.2 ms is
   still worse than XNNPACK's 19.0 ms once the 6.7 ms copy is added.
4. **Multi-step makes the NPU worse** (24.4 -> 29.5 -> 28.2 ms/step; compute rises
   17.2 -> 26.9 -> 26.8). The exact opposite of the GPU, where batching amortized
   the upload. Keep N = 1 on the NPU.
5. **fp32 compute (GPU32) is strictly worse** than fp16 on every axis (66.9 vs
   50.8 ms/step, readback 52.7 vs 37.1).

## B. End-to-end per utterance

28 prompt + 70 generated frames, single session (M5). `dec_tx`/`seanet` are the
Mimi decoder-transformer and SEANet decoder stages.

| scenario | lm / dectx / dec | LM steps/inv | RTF | total ms | LM in / run / read ms | dec_tx ms | seanet ms |
|---|---|---|---|---|---|---|---|
| `gold_cpu` | CPU/CPU/CPU | 1 | 0.81x | 7292 | 143 / 1858 / 17 | 488 | 4634 |
| `lm_npu` | NPU/CPU/GPU | 1 | 1.29x | 4526 | 654 / 1777 / 55 | 547 | 1389 |
| `lm_npu_ms4` | NPU/CPU/GPU | 4 | 1.18x | 4853 | 331 / 2442 / 79 | 523 | 1368 |
| `lm_npu_ms8` | NPU/CPU/GPU | 8 | 1.21x | 4687 | 225 / 2390 / 33 | 547 | 1386 |
| `lm_npu_ms4_dectx_npu` | NPU/NPU/GPU | 4 | 1.31x | 4327 | 310 / 2362 / 67 | 117 | 1375 |
| **`lm_cpu_dectx_npu`** | CPU/NPU/GPU | 1 | **1.53x** | **3677** | 164 / 1937 / 12 | **131** | 1362 |
| `lm_cpu_dec_gpu` | CPU/CPU/GPU | 1 | 1.37x | 4138 | 159 / 2027 / 11 | 489 | 1364 |
| `lm_cpu_all_gpu` | CPU/GPU/GPU | 1 | 1.49x | 3815 | 155 / 2028 / 13 | 142 | 1375 |
| `shipped_gpu_lm` | GPU/CPU/GPU | 1 | 0.79x | 7113 | 1195 / 115 / 3765 | 556 | 1350 |
| `lm_gpu_ms4` | GPU/CPU/GPU | 4 | 0.97x | 5894 | 619 / 70 / 3181 | 546 | 1366 |
| `lm_gpu_ms8` | GPU/CPU/GPU | 8 | 1.01x | 5596 | 424 / 64 / 2994 | 551 | 1385 |
| `lm_gpu32` | GPU32/CPU/GPU | 1 | 0.65x | 8709 | 1287 / 129 / 5252 | 536 | 1353 |
| `all_gpu` | GPU/GPU/GPU | 1 | 0.86x | 6532 | 1134 / 122 / 3654 | 135 | 1357 |

Table A predicts these: `lm_gpu_ms8` ~= 70 x 29.5 + 28 x 50.8 = 3486 ms, measured
424 + 64 + 2994 = 3482 ms.

The best placement on this device is **`lm:CPU dectx:NPU dec:GPU` at 1.53x**, ahead
of the previous best `lm_cpu_all_gpu` (1.49x) and the shipped `lm_cpu_dec_gpu`
(1.37x). Note that the win comes entirely from `dec_tx` (131 ms vs 489 ms on CPU),
not from the LM.

## C. Prompt prefill (GPU)

The text prompt is not autoregressive, so it can be run as one parallel causal pass
instead of one invocation per token. 128 synthetic tokens:

| prompt path | invocations | copy-in ms/token | compute ms/token | readback+sync ms/token | ms/token | total ms |
|---|---|---|---|---|---|---|
| per-token 1-step (GPU) | 128 | 11.89 | 1.27 | 36.47 | 49.6 | 6352 |
| per-token 1-step (CPU) | 128 | 1.43 | 19.52 | 0.13 | 21.1 | 2697 |
| `pf32` (32-wide pass) | 4 | 0.38 | 0.02 | 2.04 | 2.4 | **312** |
| `pf64` (64-wide pass) | 2 | 0.23 | 0.01 | 1.06 | 1.3 | **167** |

End-to-end with prefill (same session):

| scenario | LM steps/inv | RTF | LM in / run / read ms |
|---|---|---|---|
| `lm_gpu_ms4` | 4 | 0.98x | 565 / 61 / 3133 |
| `lm_gpu_ms4_pf32` | 4 + pf32 | **1.29x** | 283 / 39 / 2111 |
| `lm_gpu_ms8` | 8 | 1.09x | 383 / 65 / 2886 |
| `lm_gpu_ms8_pf64` | 8 + pf64 | **1.38x** | 99 / 28 / 2019 |
| `lm_gpu_pf32` | 1 + pf32 | 1.02x | 810 / 92 / 2648 |

Prefill is a **20x prompt speedup** (49.6 -> 2.4 ms/token) for ~1.3 s of extra graph
load, and it turns the GPU LM from a loss (0.98x) into a win (1.29x) end to end.

## D. Decoder stages

Same M5 session:

| dectx | dec | dectx ms | seanet ms |
|---|---|---|---|
| CPU | CPU | 488 | 4634 |
| CPU | GPU | 489 | 1364 |
| GPU | GPU | 135-142 | 1357-1375 |
| **NPU** | GPU | **117-131** | 1362-1375 |

The NPU decoder-transformer is the one accelerator that beats CPU on this stage by
a wide margin (131 vs 489 ms, ~3.7x) and it is also the placement that is faithful
to the CPU reference. The GPU delegate is as fast but changes the decoder's output
audibly, which is why `dectx` defaults to CPU on every device except a Tensor G5
with both the AOT graph and the dispatch shim installed.

## E. int8 flow-LM on CPU (M6)

Weight precision, not FLOPs, is the flow-LM's lever on CPU: the step is
DRAM-bandwidth-bound (84.44M weights, batch-1 activations), so int8 weights cut
the per-frame traffic 4x. Dynamic-range int8 keeps the graph I/O fp32, so the
host protocol is unchanged.

| LM graph | accel | copy-in ms/step | compute ms/step | readback ms/step | total ms/step | size |
|---|---|---|---|---|---|---|
| `fused_fp16` | CPU | 1.33 | 19.59 | 0.13 | 21.1 | 161.4 MB |
| **`dyn8_all`** | CPU | 1.50 | **8.65** | 0.12 | **10.3** | 81.7 MB |
| `dyn8_body` | CPU | 1.45 | 9.86 | 0.12 | 11.4 | 105.6 MB |
| `dyn4_all` | CPU | 1.68 | 7.09 | 0.13 | 8.9 | 45.6 MB |
| `wo8_all` | CPU | 1.39 | 18.90 | 0.13 | 20.4 | 81.7 MB |

`wo8_all` is weight-only quantization: the same 82 MB file, but it inserts
`DEQUANTIZE` and computes in float, so it saves nothing (1.04x). That is the
control proving the cost is bandwidth, not arithmetic.

End to end at `lm:CPU dectx:NPU dec:GPU`: 1.51x fp16 -> **2.16x** `dyn8_all`
(3680 -> 2564 ms). All-CPU: 0.83x -> 0.99x.

Full-graph static int8 turns all 7 inputs and the output int8 and collapses
(latent corr -0.002 vs eager); int16 makes the I/O int16; FC-only static keeps
fp32 I/O but coarsens activations and stops generation after 3 frames. The exact
recipe, the rejected variants and the caveats: `int8_lm.md`.

## What to keep

- **int8 weights for the flow-LM** (`pt_flowlm_fused_dyn8_all.tflite`) — 2.27x on
  the LM step, 2.16x end to end, half the size, and accepted by ear. The app
  prefers it when it has been pushed. See `int8_lm.md`.
- **LM on CPU, decoder-transformer on NPU, SEANet on GPU** (`1.53x` fp16, `2.16x`
  with int8 — the best measured placement). `Placement.default` opts into the NPU
  `dectx` on a Pixel-class GPU when `pt_mimi_dec_tx_fp16_g5.tflite` and
  `libLiteRtDispatch_GoogleTensor.so` are both present.
- **N = 1 LM steps on the NPU, N = 4-8 on the GPU.** The two accelerators want
  opposite things: the GPU is transfer-bound and rewards batching, the NPU is
  compute-bound and is penalized by it.
- **Prefill for the prompt** on GPU; it is the largest single win measured
  (49.6 -> 2.4 ms/token).
- **Do not** put the flow-LM on the GPU on PowerVR (0.79-1.01x), and do not use
  fp32 compute anywhere (GPU32 is the slowest row in every table).

By ear, every WAV in this sweep was acceptable; the audio exhibits are in
`bench/audio/`. That makes timing the deciding axis for placement here, with the
precision differences showing up as a different-but-plausible realization rather
than as noise or artefacts.

## Sources

| report | branch | covers |
|---|---|---|
| `bench/2026-09-19-pixel10-optim-multistep_lm-bdd161a.txt` | `optim/multistep_lm` | M0 baseline, single-step placements |
| `bench/2026-09-19-pixel10-optim-multistep-d2ac47a.txt` | `optim/multistep` | M2 GPU multi-step (own session) |
| `bench/2026-09-19-pixel10-optim-litertlm-1b541df.txt` | `optim/litertlm` | M3 prompt prefill (section C) |
| `bench/2026-09-19-pixel10-optim-tensor_g5-npu-singlestep.txt` | `optim/tensor_g5` | M4 NPU single-step (own session) |
| `bench/2026-09-19-pixel10-optim-tensor_g5-npu-multistep.txt` | `optim/tensor_g5` | M5 NPU multi-step (sections A, B, D) |
| `bench/2026-09-19-pixel10-optim-tensor_g5_int8-4038d13.txt` | `optim/tensor_g5_int8` | M6 int8 flow-LM (section E) |

Sections A, B and D come from one session and are comparable with each other;
section C is its own session, M2/M4 are separate sessions again, and section E is
the M6 session. Do not mix numbers across sections.

Design notes, the branch topology and the per-milestone reasoning live in
`multistep_lm_plan.md`; the int8 decision and recipe live in `int8_lm.md`.
