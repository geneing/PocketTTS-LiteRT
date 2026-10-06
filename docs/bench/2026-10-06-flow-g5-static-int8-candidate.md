# FlowLM static W8/A8 for Tensor G5: host rejection

This isolated option uses the position-major fp32 FlowLM source graph and AI
Edge Quantizer 0.8.0's built-in `static_wi8_ai8` recipe. It follows Google's
[AEQ migration example](https://developers.google.com/edge/litert/quantization/tflq_to_aeq_migration)
by disabling quantization on `INPUT` and `OUTPUT` operations, preserving the
Android float32 external protocol. Google recommends calibrated static
quantization for NPU deployment in its
[model optimization guide](https://developers.google.com/edge/litert/quantization/model_optimization).

## Reproduction

Run in the pinned WSL conversion environment with
`PYTHONPATH=/home/eingerman/pocket-tts-ref` and `PT_OUT` pointing to an isolated
output directory containing `pt_flowlm_fused_fp32_contiguous.tflite`:

```bash
python scripts/quantize_flowlm_g5.py --recipe static8 \
  --calibration-voices alba --calibration-seed 11 \
  --calibration-samples 8 --calibration-run 96 --steps 32 \
  --compare-dyn8 /path/to/pt_flowlm_fused_dyn8_all.tflite
```

Calibration free-ran eager fp32 from Alba's pinned voice K/V state at offset
126 to 222, retaining eight samples at offsets 126, 138, 150, 162, 174, 186,
198, and 210. Samples included position-specific RoPE and masks, current K/V
state, and generated inputs/noise. K/V samples were transposed to
position-major order. The script checks all sample shapes and float32 dtypes
before invoking AEQ profiler-based calibration. The voice list, seed, sample
count, and run length are command-line options.

The result has seven float32 inputs and one float32 packed output. Its K/V
inputs are each `[1,512,96,64]`; its output is `[1,12321]` for EOS, latent,
new K, and new V. The Interpreter reports 824 int8 tensors, 55 `QUANTIZE` ops,
and one `DEQUANTIZE` op. The model is therefore internally quantized while the
existing Android float cache interface survives.

## Host quality

The script ran 32 autoregressive steps from the same Alba state and noise for
static W8/A8, eager fp32, and the shipped group-major CPU dyn8 graph. Each
candidate fed its own latents and emitted K/V rows back on later steps, so
these figures include recurrent drift. The first four steps isolate the
immediate graph error before the dyn8 trajectory also diverges over a long
free run:

| Step | Static latent correlation | CPU dyn8 correlation | Static EOS abs delta | CPU dyn8 EOS abs delta | Static K/V max abs delta | CPU dyn8 K/V max abs delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.01050749 | 0.99981317 | 6.465 | 0.6556 | 4.769 | 0.09495 |
| 1 | -0.10552931 | 0.99980694 | 11.34 | 0.2444 | 4.466 | 0.1647 |
| 2 | 0.03417506 | 0.99961405 | 8.495 | 0.2223 | 4.989 | 0.2729 |
| 3 | 0.00221158 | 0.99950914 | 9.594 | 0.1625 | 5.465 | 0.3640 |

Across all 32 steps, static W8/A8 minimum latent correlation was -0.17439810,
maximum latent absolute delta 6.130, maximum EOS absolute delta 11.34, and
maximum emitted K/V absolute delta 7.686. The CPU dyn8 baseline's corresponding
values were -0.12466197, 6.044, 2.781, and 4.727. The free-run trajectories
eventually diverge for CPU dyn8 as well, but static W8/A8 fails on the *first*
step at an offset represented directly in calibration. It is not suitable for
speech generation or a Pixel latency comparison.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `pt_flowlm_fused_fp32_contiguous.tflite` | 338,170,404 | `908a5c9f9487d5ba44b6fe4e0d79f8921a626781fdf222a5c5e08ee5267326fa` |
| `pt_flowlm_fused_st8_floatio_contiguous.tflite` | 85,856,048 | `fcbd4b75b50d1af575db848bd0afa35131c76d73e508c66bb86fd09d3138d956` |

No Tensor G5 AOT compile, offload report, or device probe was run. The candidate
failed host quality, which was the requested stop condition before AOT and
device work. The attached Pixel remained untouched. This option is **not ready**
for paired long Pixel testing.
