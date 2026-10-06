# FlowLM dynamic INT8 on Tensor G5: opt-in candidates

This is a host, AOT, and initial Pixel checkpoint. The production graph remains
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
compiled opt-in candidate; its Pixel quality and latency are unmeasured here.

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

The Pixel long-utterance harness run supplied by the parent task for the
**group-major dynamic INT8 `no_truncation` AOT** failed its latency gate:
`753` candidate frames versus `759` CPU-control frames, and `52.38 s` versus
`24.76 s` end-to-end (`2.11×` slower). The reverse pair was not run. This
result applies to that 199.9 MB, two-signature group-major artifact; it does
not measure the single-signature position-major candidate or the G5 compiler's
`half` truncation setting.

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
`npuPositionMajorCache=false`. The group-major `no_truncation` artifact already
failed the Pixel latency gate; the position-major `half` artifact still needs
the Pixel A/B quality and latency protocol before adoption.

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
