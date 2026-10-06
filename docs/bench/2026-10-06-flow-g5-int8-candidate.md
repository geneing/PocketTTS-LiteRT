# FlowLM dynamic INT8 on Tensor G5: opt-in candidates

This is a host and AOT checkpoint, not a Pixel quality or latency result. The
production graph remains `pt_flowlm_fused_dyn8_all.tflite` on CPU by default.

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
dynamic INT8 graph bit-for-bit for latent and K/V output; the graph therefore
preserves the shipped quantized step across the layout change on the host.

The short free run against eager fp32 gave minimum 32-dimensional latent
correlation `0.99950914`, maximum latent absolute difference `0.06940`, and
maximum K/V absolute difference `0.3640` by step 4. The first-step EOS logit
difference was `0.6556`; later EOS differences were `0.2444`, `0.2223`, and
`0.1625`. This establishes float interface and short-rollout behavior, not
speech quality, EOS stopping agreement, or Pixel latency. No device test or
long-utterance harness was run here.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `pt_flowlm_fused_dyn8_all.tflite` | 87,504,288 | `895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9` |
| `pt_flowlm_fused_dyn8_all_no_truncation_g5.tflite` | 199,891,664 | `0d7e9a962705a453b792cd90b1816b671f394dce4c172a53ca1a81d82380545e` |
| `pt_flowlm_fused_fp32_contiguous.tflite` | 338,170,404 | `908a5c9f9487d5ba44b6fe4e0d79f8921a626781fdf222a5c5e08ee5267326fa` |
| `pt_flowlm_fused_dyn8_all_contiguous.tflite` | 85,707,824 | `0859169d1db2607512cca9f8a3591afdc0d5bbe1b6b3eb475c109f44d2f6abca` |

The 199.9 MB group-major AOT file is larger than its 87.5 MB dynamic INT8
source and the 172.4 MB fp16 AOT file. LiteRT's public Interpreter exposes only
a `DISPATCH_OP` with float32 external tensors in the compiled model; the vendor
program and packed weights are opaque. These observations do **not** establish
whether the Tensor compiler retained or expanded int8 weights. The large
group-major AOT file also includes the unused prefill signature. The
single-signature position-major variant avoids that extra compiled subgraph.

## Candidate use after AOT validation

Once the position-major source is AOT-compiled, the output should be named
`pt_flowlm_fused_dyn8_all_contiguous_no_truncation_g5.tflite`. The harness's
logical `npuGraph` should then be
`pt_flowlm_fused_dyn8_all_contiguous_no_truncation.tflite`, with
`npuSliceCache=true` and `npuPositionMajorCache=true`; `PocketTts.g5Variant`
adds `_g5` to the logical name. The existing group-major candidate uses logical
name `pt_flowlm_fused_dyn8_all_no_truncation.tflite` and
`npuPositionMajorCache=false`. Both remain opt-in and need the Pixel A/B quality
and latency protocol before adoption.

## Static W8/A16 fallback inspection

`build_pockettts.py` already defines `st16_all`: static range,
`ALL_SUPPORTED`, channelwise W8 and symmetric A16 using
`MIN_MAX_UNIFORM_QUANT`. The pinned AI Edge Quantizer
`RecipeManager.add_static_config(regex, operation_name,
activation_num_bits=16, weight_num_bits=8, ...)` requires representative
samples; `quant_calibration` supplies free-run step samples from a voice cache.
For the position-major graph, each sample's `args_4`/`args_5` cache must be
transposed from `[1,96,512,64]` to `[1,512,96,64]`. The config targets
supported operations rather than forcing graph input/output quantization; the
existing export script expects external float32 I/O. A generated static graph
must still be inspected before Android use. If K/V become int16, the current
`TensorBuffer.writeFloat`, native cache patch bridge, and `readFloat` protocol
would not be compatible without app/JNI changes.
