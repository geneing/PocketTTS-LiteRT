# FlowLM static W8/A16 on Tensor G5: host rejection

This isolated option applies AI Edge Quantizer 0.8.0's calibrated
`static_wi8_ai16` recipe to the position-major FP32 FlowLM source. A second
candidate quantizes only `FULLY_CONNECTED` operations to W8/A16. Both leave
the Android-facing inputs and output float32 by disabling quantization on
`INPUT` and `OUTPUT`. The models were exported and checked on the Linux host;
neither was AOT compiled or run on the Pixel because neither passed the
required tensor-level eager-versus-LiteRT parity gate.

## Reproduce

Use branch `codex/flowlm-g5-static-w8a16` and the pinned WSL environment in
`AGENTS.md`. The source graph is `pt_flowlm_fused_fp32_contiguous.tflite` in
the selected `PT_OUT` directory. The graph and the shipped group-major CPU
dyn8 baseline are present as symlinks under this worktree's ignored
`scripts/out/`; generated candidates remain there and are not Git artifacts.
Observed host versions: PyTorch 2.12.1+cpu, LiteRT Torch 0.9.3, AI Edge
LiteRT 2.1.6, and AI Edge Quantizer 0.8.0. There was no Tensor G5 AOT compiler
or Android runtime involved in this host check.

```bash
PYTHONPATH=/path/to/references/pocket-tts PT_OUT=/path/to/scripts/out \
  python scripts/quantize_flowlm_g5.py --recipe static16_floatio \
  --calibration-voices alba --calibration-seed 11 \
  --calibration-samples 8 --calibration-run 96 --steps 32

PYTHONPATH=/path/to/references/pocket-tts PT_OUT=/path/to/scripts/out \
  python scripts/quantize_flowlm_g5.py --recipe static16_fc_floatio \
  --calibration-voices alba --calibration-seed 11 \
  --calibration-samples 8 --calibration-run 96 --steps 32 \
  --compare-dyn8 /path/to/pt_flowlm_fused_dyn8_all.tflite
```

Calibration free-ran eager FP32 from Alba's pinned voice K/V state, offsets
126 to 222. It retained eight samples at offsets 126, 138, 150, 162, 174,
186, 198, and 210, including their RoPE, masks, latents, noise, and current
K/V buffers. Cache samples were transposed to position-major order and every
sample shape and dtype was checked against the FP32 source before profiler
calibration. The host rollout used a separate fixed noise seed 3 and fed each
candidate's own latent and K/V rows back on later steps.

## Interface and quantization

Both models retain seven float32 inputs in this order: embedding `[1,1,1024]`,
RoPE cosine `[1,1,1,64]`, RoPE sine `[1,1,1,64]`, mask `[1,16,1,513]`,
position-major key cache `[1,512,96,64]`, matching value cache, and noise
`[1,32]`. The one float32 output is `[1,12321]`, packed as EOS, 32 latent
values, 6144 new key values, and 6144 new value values. This is the existing
`npuPositionMajorCache=true` host protocol.

| Candidate | FP32 tensors | INT8 tensors | INT16 tensors | QUANTIZE / DEQUANTIZE ops |
|---|---:|---:|---:|---:|
| All supported operations W8/A16 | 8 | 47 | 774 | 52 / 1 |
| Fully connected operations W8/A16 | 729 | 47 | 87 | 40 / 47 |

The 47 INT8 tensors are weights. The all-operations recipe quantizes attention
and surrounding arithmetic; the narrower recipe inserts float/INT16 boundaries
around the fully connected operations. Interface preservation is therefore
possible with this quantizer, but interface preservation alone does not establish
numerical correctness or Tensor G5 speed.

## Host quality

The table compares each candidate with eager FP32 at the same step. The first
step is especially informative because there is no prior autoregressive error.

| Step | All-op latent corr | FC-only latent corr | All-op EOS abs delta | FC-only EOS abs delta | All-op K/V max abs delta | FC-only K/V max abs delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.07358306 | 0.94972538 | 6.485 | 0.4264 | 3.651 | 0.1334 |
| 1 | 0.03358453 | 0.95111818 | 7.016 | 5.147 | 5.522 | 1.380 |
| 2 | 0.03316675 | 0.88852207 | 10.53 | 4.327 | 5.373 | 2.221 |
| 3 | 0.05286429 | 0.89682399 | 6.861 | 3.406 | 6.063 | 2.717 |

| 32-step result | All supported operations | Fully connected only |
|---|---:|---:|
| Minimum latent correlation | -0.17160299 | 0.01055975 |
| Maximum latent absolute delta | 6.410 | 4.478 |
| Maximum EOS absolute delta | 11.15 | 5.980 |
| Maximum emitted K/V absolute delta | 7.707 | 4.387 |

The CPU dyn8 graph's first-step latent correlation in the paired static W8/A8
report was 0.99981317, with EOS delta 0.6556 and K/V delta 0.09495. The
corrected group-major step-signature runner reproduced that report's 32-step
CPU dyn8 summary here: minimum latent correlation -0.12466197, maximum latent
delta 6.044, maximum EOS delta 2.781, and maximum K/V delta 4.727. Long free
runs can diverge for the CPU baseline as well; the all-ops W8/A16 candidate
already fails at step zero, and the FC-only candidate differs substantially
before recursive drift.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `pt_flowlm_fused_fp32_contiguous.tflite` source | 338,170,404 | `908a5c9f9487d5ba44b6fe4e0d79f8921a626781fdf222a5c5e08ee5267326fa` |
| `pt_flowlm_fused_st16_floatio_contiguous.tflite` | 85,958,016 | `7c8783d666d3173a4b53a81c3fd0485194cda385fe5e7acdc9c9a60c68359f42` |
| `pt_flowlm_fused_st16_fc_floatio_contiguous.tflite` | 86,008,704 | `a0cb92a1ac799f326e1a193c6feebaec2db4dd92976539c165638bf559695276` |

Both candidates remain **host rejected**. No Tensor G5 partition report,
device latency, energy, or listening result exists for them. The attached
Pixel was untouched by this branch. A useful next quantization attempt would
need a better calibrated or mixed precision recipe and first-step output near
the established FP32/dyn8 agreement before spending time on AOT or device runs.
