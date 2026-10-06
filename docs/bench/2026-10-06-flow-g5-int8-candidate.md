# FlowLM dynamic INT8 on Tensor G5: opt-in candidates

This is a host, AOT, and Pixel checkpoint. The production graph remains
`pt_flowlm_fused_dyn8_all.tflite` on CPU by default.

## Reproduction and interfaces

The shipped group-major dynamic-range INT8 source already exists. In the pinned
LiteRT 2.2.0 AOT environment, the following command compiled it:

```bash
.venv-aot/bin/python scripts/aot_tensor_g5.py \
  pt_flowlm_fused_dyn8_all --truncation no_truncation
```

The source has two signatures. The compiler reported `Subgraph 0 fully
compiled: 7147 / 7147 ops offloaded to 1 partitions` for the unused batched
prefill and `Subgraph 1 fully compiled: 668 / 668 ops offloaded to 1 partitions`
for the one-step graph. It took 1770.6 seconds on this WSL host. The Android
engine creates the one-step input/output buffers at signature index 1 and runs
that index; K/V are float32 group-major `[1,96,512,64]`. The existing NPU slice
cache path can use this artifact with `npuPositionMajorCache=false`.

The position-major dynamic INT8 source was subsequently compiled with:

```bash
.venv-aot/bin/python scripts/aot_tensor_g5.py \
  pt_flowlm_fused_dyn8_all_contiguous --truncation half
```

This took **21.9 seconds** and reported `Subgraph 0 fully compiled: 670 / 670
ops offloaded to 1 partitions`. The AOT model has one `serving_default`
signature, one `DISPATCH_OP`, float32 external I/O, and position-major K/V
inputs `[1,512,96,64]` each. The position-major AOT file is a separately
compiled opt-in candidate; its Pixel result is recorded below.

The new position-major source is generated from the already-exported fp32
position-major graph with the existing `dyn8_all` recipe:

```bash
.venv/bin/python scripts/quantize_flowlm_g5.py --steps 4
```

That recipe is AI Edge Quantizer 0.8.0 dynamic-range W8/float-activation,
channelwise min/max, `ALL_SUPPORTED` operation scope. On the single-signature
position-major graph, it quantized 47 `FULLY_CONNECTED` weight tensors to int8;
all seven inputs and the packed output remain float32. K/V inputs are each
`[1,512,96,64]`, matching `npuPositionMajorCache=true`. The script checks the
input/output shapes and dtypes and performs a four-step eager-fp32 rollout.
The first position-major dynamic INT8 step matched the shipped group-major
dynamic INT8 graph bit-for-bit across the full packed output (EOS, latent,
K/V; max absolute difference `0`); the graph therefore
preserves the shipped quantized step across the layout change on the host.

The short free run against eager fp32 gave minimum 32-dimensional latent
correlation `0.99950914`, maximum latent absolute difference `0.06940`, and
maximum K/V absolute difference `0.3640` by step 4. The first-step EOS logit
difference was `0.6556`; later EOS differences were `0.2444`, `0.2223`, and
`0.1625`. This establishes float interface and short-rollout behavior, not
speech quality or EOS stopping agreement. No device test or long-utterance
harness was run by the exporter.

The Pixel long-utterance harness runs supplied by the parent task failed their
latency and completion gates:

| AOT candidate | Order | Frames, NPU / CPU | End-to-end, NPU / CPU | Run stage, NPU / CPU | First audio, NPU / CPU |
|---|---|---:|---:|---:|---:|
| Group-major dynamic INT8 `no_truncation` | NPU → CPU | 753 / 759 | 52.380 / 24.764 s (`2.11×`) | 36.345 / 14.962 s (`2.43×`) | 2.901 / 1.143 s (`2.54×`) |
| Group-major dynamic INT8 `no_truncation` | CPU → NPU | 753 / 759 | 52.068 / 25.278 s (`2.06×`) | 36.503 / 15.268 s (`2.39×`) | 2.831 / 1.148 s (`2.47×`) |
| Group-major dynamic INT8 `no_truncation` | Mean | 753 / 759 | 52.224 / 25.021 s (`2.09×`) | 36.424 / 15.115 s (`2.41×`) | 2.866 / 1.146 s (`2.50×`) |
| Position-major dynamic INT8 `half` | NPU → CPU | 753 / 759 | 53.008 / 24.933 s (`2.13×`) | 37.099 / 14.951 s (`2.48×`) | 2.883 / 1.105 s (`2.61×`) |
| Position-major dynamic INT8 `half` | CPU → NPU | 753 / 759 | 53.139 / 25.479 s (`2.09×`) | 37.127 / 15.448 s (`2.40×`) | 2.846 / 1.136 s (`2.51×`) |
| Position-major dynamic INT8 `half` | Mean | 753 / 759 | 53.074 / 25.206 s (`2.11×`) | 37.113 / 15.200 s (`2.44×`) | 2.865 / 1.121 s (`2.56×`) |

The reversed pairs confirm the roughly 2.1× slowdown is not explained by run
order. Both NPU graphs stopped six frames before CPU in both orders, so they
also fail the completion gate. Position-major layout and `half` truncation did
not restore a latency advantage over CPU INT8. Their long free-running
waveform correlations were 0.115/0.081 for the group-major/position-major
CPU-first pair; these are diagnostics, not standalone speech-quality verdicts.
The WAVs for the reversed position-major pair are preserved under ignored
`build/flowlm-g5-int8/dyn8-reversed-position-major-device/`.

The same position-major INT8 artifact was also run with native Tensor G5
`HIGH_PERFORMANCE` mode enabled. Across both orders it averaged 51.919 s NPU
versus 24.793 s CPU (2.094×), with 753/759 frames; this is only about 2.2%
faster than its default-mode 53.074 s NPU mean and still fails speed and
completion. See the [runtime-mode report](2026-10-06-g5-high-performance.md)
for stage timings and paired reports/WAVs.

The stage timings identify where the measured gap occurs: each NPU LM
`CompiledModel.run()` total was 36.3-37.1 s, versus about 15.0 s for the CPU
int8 graph. In the two group-major orders, NPU host input transfer averaged
0.964 s versus 2.397 s on CPU; native cache-row patch was 2.28-2.50 s and
packed-output readback 2.87-3.13 s. So cache input transfer is not the source
of the INT8 slowdown; graph execution dominates. All graph ops were compiled
into one opaque G5 dispatch partition. Without vendor dispatch profiling,
these results do not isolate whether the internal cost comes from INT8 weight
handling, kernel selection, or another compiler/runtime choice.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `pt_flowlm_fused_dyn8_all.tflite` | 87,504,288 | `895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9` |
| `pt_flowlm_fused_dyn8_all_no_truncation_g5.tflite` | 199,891,664 | `0d7e9a962705a453b792cd90b1816b671f394dce4c172a53ca1a81d82380545e` |
| `pt_flowlm_fused_fp32_contiguous.tflite` | 338,170,404 | `908a5c9f9487d5ba44b6fe4e0d79f8921a626781fdf222a5c5e08ee5267326fa` |
| `pt_flowlm_fused_dyn8_all_contiguous.tflite` | 85,707,824 | `0859169d1db2607512cca9f8a3591afdc0d5bbe1b6b3eb475c109f44d2f6abca` |
| `pt_flowlm_fused_dyn8_all_contiguous_half_g5.tflite` | 88,265,536 | `bc8bd5c19495958dfc5b77915dea314c38a9127b839fd357e076d9038cb56acd` |
| `pt_flowlm_fused_st16_all_contiguous.tflite` | 85,957,728 | `b89ac1662f01bbbe3f2f26a71582116f9b8b867f3f404b611929e68deb61ed39` |

The 199.9 MB group-major AOT file is larger than its 87.5 MB dynamic INT8
source and the 172.4 MB fp16 AOT file. LiteRT's public Interpreter exposes only
a `DISPATCH_OP` with float32 external tensors in the compiled model; the vendor
program and packed weights are opaque. The single-signature position-major
`half` AOT file is only 2,557,712 bytes larger than its dynamic INT8 source,
which is consistent with compact weights. These observations do **not** prove
whether the Tensor compiler retained int8 weights internally. The large
group-major AOT file also includes the unused prefill signature, while the
position-major variant has only a one-step compiled subgraph. The two AOT
artifacts also used different truncation settings, so their size difference
cannot be attributed to either cause alone.

## Candidate use in the Pixel harness

The position-major AOT output is
`pt_flowlm_fused_dyn8_all_contiguous_half_g5.tflite`. Its harness logical
`npuGraph` is `pt_flowlm_fused_dyn8_all_contiguous_half.tflite`, with
`npuSliceCache=true` and `npuPositionMajorCache=true`; `PocketTts.g5Variant`
adds `_g5` to the logical name. The existing group-major candidate uses logical
name `pt_flowlm_fused_dyn8_all_no_truncation.tflite` and
`npuPositionMajorCache=false`. Both candidates failed the Pixel latency gate;
neither is selected by the production default.

## Static W8/A16 fallback inspection

`build_pockettts.py` already defines `st16_all`: static range,
`ALL_SUPPORTED`, channelwise W8 and symmetric A16 using
`MIN_MAX_UNIFORM_QUANT`. It matches the pinned AI Edge Quantizer 0.8.0 built-in
`recipe.static_wi8_ai16()` configuration. The pinned default policy covers
`FULLY_CONNECTED`, `BATCH_MATMUL`, attention arithmetic and shape ops, and
`INPUT`/`OUTPUT`. `quant_calibration` supplies free-run step samples from a
voice cache; each sample's K/V was transposed from `[1,96,512,64]` to
`[1,512,96,64]`. This experiment used eight samples over 32 steps:

```bash
.venv/bin/python scripts/quantize_flowlm_g5.py --recipe static16 \
  --calibration-samples 8 --calibration-run 32 --steps 4
```

The resulting graph has 47 int8 weight tensors, int16 activations, and 48
`QUANTIZE` ops. **All seven inputs and the packed output are int16**, including
both full K/V caches. K and V external quantization scales differ (about
`2.94e-4` and `4.38e-5`). The current Android `writeFloat`/`readFloat` and
native float cache-row patch cannot use it. Python LiteRT `TensorBuffer.write`
also rejects int16, so the host rollout used `Interpreter.set_tensor` and
dequantized its output.

Its four-step eager-fp32 latent correlations were `-0.0565`, `-0.0992`,
`-0.0357`, and `-0.1376`; the maximum latent absolute difference was `6.401`
and maximum K/V difference `5.900`. EOS logit differences ranged from `6.37`
to `10.42`. This candidate fails the host quality proxy as well as the Android
cache interface. No G5 AOT compile of the static graph is warranted.

A follow-up on branch `codex/flowlm-g5-static-w8a16` explicitly disabled
quantization for INPUT/OUTPUT, preserving the app's float32 interface. Its
all-supported-ops variant still failed at step zero (latent correlation
-0.0736); an FC-only variant began at 0.9497 but fell to 0.0106 minimum over
32 autoregressive steps. Neither passed the host parity gate, so neither was
AOT compiled. See the [float-interface W8/A16 report](2026-10-06-flow-g5-static-w8a16-candidate.md).
