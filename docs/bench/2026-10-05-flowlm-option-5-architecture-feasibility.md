# FlowLM option 5 architecture feasibility

Date: 2026-10-05

Status: stop before architecture changes, training, or device benchmarking. The required training and held-out quality gates cannot pass with the assets and tooling in this checkout.

## Gate from the research brief

Option 5 proposes GQA/MQA, a smaller transformer or FFN, local-window/voice-memory changes, or a smaller flow head. The brief requires a trained or distilled candidate and a held-out speech suite covering voices, text lengths, numerals, punctuation, and long passages. Acceptance includes intelligibility/transcription, completion and duration, speaker similarity, artifact listening, and repeated-seed stability. Tensor parity or slicing existing weights is not sufficient. See [`flowlm-pixel10-optimization-research.md`](../flowlm-pixel10-optimization-research.md#5-change-the-model-architecture-and-retraindistill-for-mobile-decode).

## Checkout audit

| Prerequisite | Evidence in this worktree | Result |
|---|---|---|
| Training, distillation, or QAT pipeline | `scripts/` contains the converter, AOT helper, model/voice utilities, and pack/server scripts; no model trainer or distillation entry point was found. | Missing |
| Teacher/student checkpoints | No `.safetensors`, `.pt`, `.pth`, `.ckpt`, `.onnx`, `.tflite`, or model `.bin` files were present in the checkout outside generated build metadata. `references/`, `scripts/out/`, and `.venv/` are absent. | Missing locally |
| Train/held-out speech corpus | No audio, transcript-manifest, or dataset files (`.wav`, `.flac`, `.mp3`, `.jsonl`, `.parquet`, `.arrow`, `.csv`, or `.tsv`) were present in the checkout. | Missing |
| Export path for a changed architecture | [`scripts/build_pockettts.py`](../../scripts/build_pockettts.py) loads the upstream `TTSModel` and reauthors its existing modules. Its geometry is fixed in module constants and its `FlowLMStep` copies the current projection weights into fixed 16-head shapes. It has no student-checkpoint import or architecture-variant configuration. | Existing-model export only |
| Android graph/voice packaging for another KV topology | [`PocketTts.kt`](../../pockettts-core/src/main/kotlin/dev/pockettts/PocketTts.kt) and [`create_voice.py`](../../scripts/create_voice.py) encode the current six-layer, sixteen-head cache geometry. No smaller-KV graph or compatible voice-cache converter is present. | Missing |

The exporter can obtain the upstream teacher through the pinned Pocket TTS reference package and its normal model-loading path when that environment and weights are available. That is useful for converting the current model; it is not a training source, a student checkpoint, or an export route for an altered architecture. This audit only establishes absence from this checkout, not from external caches.

## Current geometry and expected structural impact

The checked-in converter and Android constants agree on six layers, width 1024, 16 query/KV heads, head dimension 64, FFN width 4096, and 32-dimensional Mimi latents. The cache capacity is 512 positions. Each step passes separate K and V tensors shaped `[1, 96, 512, 64]` (`6 × 16` groups), in float32.

That is 25,165,824 bytes for K and V together per invocation:

`2 × 6 layers × 16 heads × 512 positions × 64 values × 4 bytes = 25,165,824 bytes`.

Four KV heads per layer would reduce this cache payload to 6,291,456 bytes, a 4× structural reduction. This is a shape-only estimate, not evidence of a trained model's quality or a measured latency/energy gain. GQA changes K/V projection and cache shapes while retaining query heads; the existing projection copy, attention indexing, output signature, voice-cache format, and host constants all assume the original topology. Layer pruning or width reduction likewise changes learned computations and cannot be validated by slicing tensors.

## What `FlowLmHarnessTest` can establish

[`FlowLmHarnessTest.kt`](../../app/src/androidTest/java/com/pockettts/FlowLmHarnessTest.kt) runs one fused FlowLM output per prompt token, starts from a real voice KV prefix, feeds zero noise, updates the returned KV row, and compares candidate tensors with a CPU fp16 graph. It records graph-load and run time plus correlation, mean absolute difference, and mean squared difference. Its tensor sizes are derived from the current `PocketTts` constants.

This is a useful step-level diagnostic once a candidate graph and matching tokenizer, embedding table, and voice cache exist. It does not generate speech or measure transcription, voice similarity, listening quality, long-form completion, repeated-seed stability, power, or sustained RTF. Those remain required for option 5. No harness run was attempted: the checkout has no model/voice/embedding artifacts, and this evaluation was assigned not to use the shared Pixel.

## Stop decision and prerequisites to resume

Do not alter the architecture or slice the teacher weights in this branch. Option 5 cannot pass its own stop gate until the project has:

1. The pinned teacher checkpoint and a reproducible training environment.
2. A licensed, versioned training corpus plus a disjoint held-out speech suite with transcripts and voice identities.
3. A selected student design and training/distillation recipe, with a saved student checkpoint and repeatable quality evaluation.
4. A shape-aware exporter and Android model/voice format for that student's exact architecture, plus step-level checks through `FlowLmHarnessTest`.
5. Held-out device runs against the production CPU dynamic-int8 control and the research brief's speed, energy, completion, listening, and stability gates.

The branch therefore records a feasibility stop, not an optimization result. No performance or quality claim is made.
