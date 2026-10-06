# FlowLM selective W8/A16: host candidates

The full static W8/A16 and all-FC W8/A16 recipes failed first-step tensor
parity (see the companion `2026-10-06-flow-g5-static-w8a16-candidate.md`).
This follow-up quantizes only large FlowLM feed-forward network (FFN) dense
matrices. It retains float32 attention projections, K/V handling, EOS and
flow head, and all Android-facing tensors. Both selective candidates passed
the first-step host screen and completed a 32-step Alba free-run comparison.
The FFN12 variant was also AOT compiled for Tensor G5 on the host. There is
still no Pixel timing, energy, speech completion, or listening result.

## Selection and reproduction

The position-major source has 47 `FULLY_CONNECTED` operations. Each of its six
FlowLM layers has one `[4096,1024]` FFN expansion matrix and one `[1024,4096]`
FFN output matrix. `static16_ffn12` quantizes both matrices in all six layers,
covering 12/47 dense operations and 50,331,648/84,444,160 dense weights
(59.6%). `static16_ffn6` quantizes only the six FFN output matrices, covering
6/47 operations and 25,165,824/84,444,160 dense weights (29.8%).

The CLI discovers the exact source output tensor names by weight shape and
restricts the AEQ recipe to those names. It requires exactly 12 or six unique
matches and verifies the exported INT8 weight count and shapes. AEQ's
`get_op_scope` adds a trailing semicolon to the output name; the exact-match
regex includes it. The first attempted selector omitted it and produced an
unquantized source copy; the coverage assertion caught that result before
parity testing. Only the corrected candidates below count as experiments.

Run from branch `codex/flowlm-g5-static-w8a16` in the pinned WSL environment
with source `pt_flowlm_fused_fp32_contiguous.tflite` in `PT_OUT`. Observed
versions: PyTorch 2.12.1+cpu, LiteRT Torch 0.9.3, AI Edge LiteRT 2.1.6,
AI Edge Quantizer 0.8.0. The local ignored `scripts/out/` contains source
and CPU dyn8 symlinks; generated artifacts are not committed.

```bash
PYTHONPATH=/path/to/references/pocket-tts PT_OUT=/path/to/scripts/out \
  python scripts/quantize_flowlm_g5.py --recipe static16_ffn12 \
  --calibration-voices alba --calibration-seed 11 \
  --calibration-samples 8 --calibration-run 96 --steps 32 \
  --compare-dyn8 /path/to/pt_flowlm_fused_dyn8_all.tflite

PYTHONPATH=/path/to/references/pocket-tts PT_OUT=/path/to/scripts/out \
  python scripts/quantize_flowlm_g5.py --recipe static16_ffn6 \
  --calibration-voices alba --calibration-seed 11 \
  --calibration-samples 8 --calibration-run 96 --steps 32 \
  --compare-dyn8 /path/to/pt_flowlm_fused_dyn8_all.tflite
```

Both use profiler calibration from Alba's pinned voice state, free-running
offsets 126..222 and retaining eight samples at 126, 138, 150, 162, 174,
186, 198 and 210. Each sample includes live embedding, RoPE, mask, K/V,
and noise inputs. The rollout starts from the same Alba voice at offset 126
and uses noise seed 3. Each graph feeds back its own latent and emitted K/V
rows after the first step; eager FP32 and CPU dyn8 each have independent
feedback loops.

## Graph interface and actual quantization

Both candidates have seven float32 inputs: embedding `[1,1,1024]`, cosine and
sine `[1,1,1,64]` each, mask `[1,16,1,513]`, position-major K/V each
`[1,512,96,64]`, and noise `[1,32]`. Their one float32 output is `[1,12321]`
for EOS, 32 latent values, 6144 new K values and 6144 new V values.
The current Android position-major host protocol therefore remains valid.

| Candidate | INT8 FC weights | INT16 tensors | Quantize / dequantize ops | Float32 FC weights |
|---|---:|---:|---:|---:|
| FFN12 | 12/47 (six each `[4096,1024]` and `[1024,4096]`) | 24 | 12 / 12 | 35/47 |
| FFN6 | 6/47 (all `[1024,4096]`) | 12 | 6 / 6 | 41/47 |

All QKV projection, attention output, EOS, and flow-head FC weights remain
float32 in both variants. The quantized matrix activation and output tensors
are INT16; external cache and result tensors remain float32.

## Host tensor quality

The first four autoregressive steps compare each graph with the eager FP32
trajectory. CPU dyn8 first-four values use the same Alba/noise setup and were
recorded in `2026-10-06-flow-g5-static-int8-candidate.md`; its complete
32-step summary was re-measured by this script using the graph's step
signature (index 1).

| Step | FFN12 latent corr | FFN6 latent corr | CPU dyn8 latent corr | FFN12 EOS abs delta | FFN6 EOS abs delta | CPU dyn8 EOS abs delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.99980462 | 0.99981426 | 0.99981317 | 0.3122 | 0.3932 | 0.6556 |
| 1 | 0.98985330 | 0.99002487 | 0.99980694 | 0.6995 | 0.5963 | 0.2444 |
| 2 | 0.95310265 | 0.95876089 | 0.99961405 | 0.8664 | 0.5801 | 0.2223 |
| 3 | 0.95751255 | 0.95882887 | 0.99950914 | 1.033 | 1.107 | 0.1625 |

| Step | FFN12 latent max delta | FFN6 latent max delta | FFN12 K/V max delta | FFN6 K/V max delta | CPU dyn8 K/V max delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.08976 | 0.08998 | 0.1021 | 0.09839 | 0.09495 |
| 1 | 0.8354 | 0.7801 | 0.5876 | 0.1487 | 0.1647 |
| 2 | 0.9927 | 0.9678 | 0.5769 | 0.5939 | 0.2729 |
| 3 | 0.7158 | 0.7544 | 0.7887 | 0.5051 | 0.3640 |

| 32-step self-fed result | FFN12 | FFN6 | CPU dyn8 |
|---|---:|---:|---:|
| Minimum latent correlation | 0.66987199 | 0.39914435 | -0.12466197 |
| Maximum latent absolute delta | 2.486 | 4.388 | 6.044 |
| Maximum EOS absolute delta | 1.931 | 4.412 | 2.781 |
| Maximum emitted K/V absolute delta | 2.663 | 2.772 | 4.727 |

The selective results show much better first-step behavior than the full or
all-FC static W8/A16 graphs. The CPU dyn8 model has closer step-1..3 latent
agreement, while its 32-step free-run diverges more in this one seed; free-run
correlation is a trajectory diagnostic, not a speech quality verdict. FFN12
retains more INT8 weight coverage and had the stronger 32-step result here.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `pt_flowlm_fused_fp32_contiguous.tflite` source | 338,170,404 | `908a5c9f9487d5ba44b6fe4e0d79f8921a626781fdf222a5c5e08ee5267326fa` |
| `pt_flowlm_fused_st16_ffn12_contiguous.tflite` | 187,545,872 | `3f045abcf256ff1716cc497c26abd2f49a4c880d2ca9c6c990a3306a3e5336f0` |
| `pt_flowlm_fused_st16_ffn6_contiguous.tflite` | 262,745,632 | `ad2ebba4147cea2253ba7c371de3c013c360aea54ae89fe3cc3cc715e3d10211` |
| `pt_flowlm_fused_st16_ffn12_contiguous_no_truncation_g5.tflite` | 178,697,824 | `183a71a2475e6fb68ab626788f8f2189a85d9fb661bcca8c8637e19ce2906d12` |

## Tensor G5 AOT result for FFN12

The separate pinned AOT environment has `ai-edge-litert==2.2.0` and
`ai-edge-litert-sdk-google-tensor==2.2.0`. Only FFN12 was compiled, using:

```bash
PT_OUT=/path/to/scripts/out .venv-aot/bin/python scripts/aot_tensor_g5.py \
  pt_flowlm_fused_st16_ffn12_contiguous --truncation no_truncation
```

The compiler completed in **34.0 s** and reported `Subgraph 0 fully
compiled: 694 / 694 ops offloaded to 1 partitions`. The exported file above
contains one `serving_default` signature and one public `DISPATCH_OP`.
Interpreter inspection found the same seven float32 inputs and one float32
`[1,12321]` output, including position-major K/V inputs `[1,512,96,64]`.
The compiled program is opaque, so the host cannot establish which internal
weight precision the vendor program ultimately uses or execute its numerical
path without Tensor G5.

The compiled artifact is at
`build/flowlm-worktrees/option12-static-w8a16/scripts/out/pt_flowlm_fused_st16_ffn12_contiguous_no_truncation_g5.tflite`.
Its logical Android `npuGraph` name is
`pt_flowlm_fused_st16_ffn12_contiguous_no_truncation.tflite` with
`npuPositionMajorCache=true`. Full speech completion, listening and paired
Pixel timing are still required by the benchmark protocol. No device work was
performed in this branch.
