# SEANet decoder on the Tensor G5 NPU — experiment (negative result)

Branch `experiment/seanet-npu`. The Mimi **decoder transformer** (`pt_mimi_dec_tx`)
already runs on the NPU in the shipped placement; this experiment asked whether the
**SEANet decoder** (`pt_mimi_deconly`) can join it. Short answer: it can be made to
compile and it is numerically fine, but it is ~500x slower than the GPU, so the
placement is **not adopted**. The working artifacts and the upstream crash report
are kept here for the record.

## 1. The blocker: `RESIZE_NEAREST_NEIGHBOR` crashes the Tensor compiler

`python scripts/aot_tensor_g5.py pt_mimi_deconly_fp16` used to fail with an opaque

```
ERROR: [compiler_plugin.cc:876] Failed to compile model:
Compilation has failed with error type: INTERNAL.
ValueError: apply_plugin failed to apply plugin. See /tmp/tmpXXXX.error for details.
```

All 167 ops partition cleanly ("Subgraph 0 fully compiled") and then the backend
compiler worker segfaults, so there is **no op-level diagnostic** in the error file
— only `gnu_build_id` and the worker's `memory_map`. The same happens for the fp32
graph, so it is not a dtype issue.

`scripts/npu_seanet.py` bisects this by exporting decoder-prefix graphs
(`MimiDecOnly.plan[:upto]`) and AOT-compiling each:

| prefix | contents | Tensor G5 |
|---|---|---|
| 0 | `conv 512->512 k7` | OK (7/7 ops) |
| 1 | + ELU | OK (13/13 ops) |
| 2 | + first `ZeroStuffConvT1d` (s6) | **FAIL** |
| …26 | everything longer | FAIL |

Isolating that single layer (`npu_seanet.py zs`), window 4096:

| variant | ops kept | result |
|---|---|---|
| `orig` | nearest resize + mask + grouped conv1d + crop | FAIL |
| `nomask` | resize + conv (mask removed) | FAIL |
| `noconv` | resize only | FAIL |
| `noslice` | resize + conv, no crop | FAIL |
| `orig` at window 512 | same layer, 8x smaller | OK (14/14) |
| `padonly` | `F.pad` instead of resize (= wrong semantics) | OK |
| `padinsert` | reshape `(B,C,L,1)` + `F.pad` + reshape | OK, **bit-identical** to `orig` |
| `convtr` | real `F.conv_transpose1d` | OK, max\|d\| 4.8e-06 |

So the culprit is the resize, not the mask, the grouped conv, the crop or the size
of the conv. `scripts/npu_crash_repro.py` is a standalone (Pocket TTS-free)
reproducer: it crashes for `[1,512,L]` fp32 resized by s=6 at L=4096 and passes at
L=2727 — i.e. the trigger is the **output tensor size** (~33.5 MB is the observed
transition on this machine), not the op itself. Nearest-resize at small sizes and
`TRANSPOSE_CONV` at the same large size both compile.

Upstream report (recommended repo: `google-ai-edge/LiteRT`, the error text itself
links its issue tracker): `NPU_crash.txt`.

## 2. Workarounds that compile

`ZeroStuffConvT1d` is the streaming-exact ConvTranspose replacement used by the
decoder, so removing the resize is the whole fix:

- `padinsert` — zero-stuff by `reshape (B,C,L,1)` → `F.pad(0, s-1)` → `reshape`,
  then the same grouped conv1d. **Bit-identical** to the current layer, adds a
  `PAD` instead of a `RESIZE_NEAREST_NEIGHBOR`. This is the recommended form.
- `convtr` — the upstream `F.conv_transpose1d` (`TRANSPOSE_CONV`). Also compiles,
  ~1e-6 off. Fewer ops (146 vs 155) but no accuracy advantage.

Both are selectable in `npu_seanet.py --zs-mode`, and both produce a decoder that
compiles at window 4096 (491520 samples) as well as 512.

## 3. Device results (Pixel 10, 20 runs)

Graph: full W512 SEANet, fp16 weights, one NPU partition
(`seanet_w512_padinsert_padinsert_fp16_aot.tflite`, 155/155 ops, 9.5 MB).

| graph | accel | avg ms | p50 ms | vs torch fp32 (corr / max\|d\| / relDb) |
|---|---|---|---|---|
| shipped `pt_mimi_deconly_w512_fp16` | GPU | 0.106 | 0.101 | 0.999921 / 6.1e-03 / −38.0 dB |
| `padinsert` SEANet | **NPU** | **57.2** | **57.1** | **0.999963 / 3.7e-03 / −41.0 dB** |
| `convtr` SEANet | NPU | 49.0 | 49.0 | 0.999963 / 3.7e-03 / −41.0 dB |

The NPU squashes the three resize stages into one partition and is numerically
*slightly better* than the fp32 GPU path (max\|d\| 3.7e-3 vs 6.1e-3). It is also
**~500x slower**, and the gap is the roofline: the decoder tail works on tensors up
to `1x64x491520`, and the NPU moves those through DRAM far more cheaply on the GPU.

Battery power could not be sampled — `/sys/class/power_supply/battery/current_now`
and `voltage_now` are not readable to the test process on this device, so the
"NPU uses less power per decode" hypothesis is untested rather than refuted.

## 4. A near/far split does not rescue it

The obvious mitigation is to keep only the short, wide-channel part on the NPU and
hand the long tensors to the GPU. The decoder is causal with a small receptive
field, so the split is exact (host-verified corr 1.0, max\|d\| 0 for both cuts), and
both halves compile to one NPU partition each. Measured on device:

| path | avg ms | p50 ms |
|---|---|---|
| all-GPU (near + far) | 0.24 | 0.25 |
| **near NPU → far GPU** | **124.2** | **124.2** |
| all-NPU | 138.4 | 155.3 |

The near stage alone is ~82x its GPU equivalent and the far stage ~464x, and the
composition is ~7 dB worse than torch. Neither the timing nor the accuracy argues
for putting any part of the SEANet on the NPU.

## 5. Verdict and what is kept

- **Keep the placement `lm:CPU dectx:NPU dec:GPU`.** The SEANet NPU variant is a
  worked export and a hardware finding, not a shippable graph.
- Keep `scripts/npu_seanet.py` (prefix/range/zs sweep + AOT), `scripts/npu_crash_repro.py`
  and `NPU_crash.txt` for the upstream report, and
  `app/src/androidTest/java/com/pockettts/SeanetNpuProbeTest.kt` as the device harness.
- If a future Tensor compiler fixes large nearest-resize, re-run
  `scripts/npu_seanet.py sweep` to re-test the plain decoder; nothing else changes.
- The `padinsert` rewrite is still worth having upstream in
  `build_pockettts.py` if the resize ever needs to go away for another backend —
  it is bit-exact and no slower on the GPU (0.106 → 0.11 ms avg, within noise).
