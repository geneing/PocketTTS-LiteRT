# Tensor G5 FP16 half-truncation candidate (host only)

This branch records an opt-in AOT variant of the existing position-major FlowLM step. It changes only the Tensor G5 compiler's float truncation setting. The source graph, quantization, Android code, and production model selection are unchanged. The generated models remain under ignored `scripts/out/` in this worktree.

## Reproduction

The existing `pt_flowlm_fused_fp16_contiguous.tflite` was copied byte-for-byte from the parent checkout into this worktree's `scripts/out/`. In WSL, using the parent's pinned `.venv-aot` (`ai-edge-litert` and `ai-edge-litert-sdk-google-tensor` 2.2.0):

```bash
PT_OUT=/mnt/i/Android_Projects/PocketTTS-LiteRT/build/flowlm-worktrees/option9-g5-half-truncation/scripts/out \
  /mnt/i/Android_Projects/PocketTTS-LiteRT/.venv-aot/bin/python \
  scripts/aot_tensor_g5.py pt_flowlm_fused_fp16_contiguous --truncation half
```

The compiler accepted `half` for `Google_Tensor_G5`, took **33.3 s**, and reported `Subgraph 0 fully compiled: 717 / 717 ops offloaded to 1 partitions`. The existing no-truncation report also records 717/717 ops in one partition; its compile duration was not captured. No compiler retry was needed.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Shared position-major FP16 source | 169,288,096 | `1d6fe48ed4435b415970a240916a726f1b8994036aacf3c009085722a8a632c7` |
| Existing `no_truncation` G5 output | 172,399,184 | `1e3142667b944172d1216728bec52bfabe3529ac6606394dee2878c0945b8d08` |
| New `half` G5 output | 172,399,184 | `fb66bc91257207b809a4aab4a77e33168cd105cd4ee266405798a4937ccfdfac` |

The candidate file is `scripts/out/pt_flowlm_fused_fp16_contiguous_half_g5.tflite` in this worktree. The copied source hash matches the parent source. Both AOT outputs have the same byte length but different hashes; a direct byte comparison found 168,545,298 differing bytes. This establishes a distinct compiler output, but the opaque vendor program does not expose which internal instructions changed.

## Tensor interface and parity scope

Host LiteRT Interpreter inspection found identical public interfaces for both compiled outputs: one `serving_default` signature, seven float32 inputs, one float32 `[1,12321]` output, and one `DISPATCH_OP`. The K and V inputs remain position-major `[1,512,96,64]`; all other input shapes match the shared source graph. The export script checks exact eager equality of the position-major layout against the original fused step, then requires first- and next-step CPU fp32 graph max absolute differences below `2e-5`. Those checks precede the FP16 conversion; they establish layout and fp32 source parity, not FP16 or compiled-program parity. The FP16 AOT input is byte-identical between these two compilations. No numerical comparison between the compiled G5 programs is available on the WSL host. AOT `DISPATCH_OP` execution and latency require the Tensor G5 device.

For context, the prior no-truncation position-major FP16 Pixel pairs averaged 32.682 s NPU inference versus 24.992 s CPU int8 inference (`1.308x` NPU/CPU) on the long Alba workload. **The half candidate has no Pixel latency, completion, or listening result yet.** It is retained as a separate opt-in artifact for a later paired device comparison; no device install or run was performed for this branch.
