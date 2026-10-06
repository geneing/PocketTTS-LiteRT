# FlowLM option 3 implementation status

Date: 2026-10-05. Device: Pixel 10 / Tensor G5 target; **no device run is
recorded by this implementation checkpoint**. The shared phone is reserved for
the serialized acceptance runs.

## Implemented prototype

- The exporter accepts `PT_FLOWLM_PMAX=128`, `256`, or `512`. The default 512
  filenames remain unchanged; reduced variants receive a `_pmaxN` suffix.
  Reduced shapes are restricted to the `fused` and `quant` build stages so an
  environment override cannot accidentally resize Mimi graphs or voice assets.
- Android `PocketTtsConfig.lmCapacity` selects the matching graph and sizes the
  engine KV, mask, and prefill buffers. Voice files keep their existing compact
  fp16-prefix format; the engine copies every live prefix row exactly into the
  selected capacity stride.
- `PocketTts.smallestFlowLmCapacity` chooses the smallest fitting bucket from
  voice length, prompt length, and the existing planned frame budget. Synthesis
  fails before prefill/generation if no bucket fits, instead of clamping away
  generated frames. Reduced-capacity NPU and multi-step paths remain gated.
- `FlowLmHarnessTest` reports graph load, input staging, run, output read, KV
  transfer bytes, tensor differences, and a deterministic zero-noise first
  decode step. Its `fp16` and `int8_per_head` cache modes are host-side
  round-trip simulations only: LiteRT graph inputs still use float32, so they
  do not reduce actual transfer bytes or represent a speed result.

For a Linux/WSL export, create an isolated `PT_OUT` directory and run the
`fused` stage followed by `quant` for each capacity. For example, capacity 256
produces `pt_flowlm_fused_fp16_pmax256.tflite` and
`pt_flowlm_fused_dyn8_all_pmax256.tflite`. Capacity 128 is exportable, but the
bundled Alba voice prefix is longer than 128; the exporter's shape/parity seed
therefore uses an empty prefix, and the Android harness must use a voice that
actually fits before that bucket is a supported candidate.

```bash
PT_FLOWLM_PMAX=256 PT_OUT=/tmp/pt-pmax256 python scripts/build_pockettts.py fused
PT_FLOWLM_PMAX=256 PT_OUT=/tmp/pt-pmax256 PT_QUANT=dyn8_all python scripts/build_pockettts.py quant
```

For app-level experimentation, set `PocketTtsConfig.lmCapacity` to the chosen
bucket; the engine adds `_pmaxN` to the configured base graph name. The focused
harness accepts the corresponding `lmGraph`, `lmCapacity`, `referenceGraph`,
`backends=CPU`, and `kvPrecision=fp32|fp16|int8_per_head` instrumentation args.
The harness defaults its CPU comparison graph to the shipped dynamic-int8
baseline. If the selected capacity graph is missing, it records the named
missing graph in `report.txt` and fails when no candidate backend completed;
the app's model store raises `FileNotFoundException` naming the absent graph.

## Acceptance status

| Check | Status |
|---|---|
| Exporter syntax | Passed with bundled Python `py_compile` |
| Capacity graph export and tensor parity | Pending; export environment/model weights not exercised here |
| Exact prefix repack / bucket planner | Added as focused `FlowLmHarnessTest` instrumentation method; not run on device |
| Android Kotlin compile | Passed: `:app:compileDebugAndroidTestKotlin` with Gradle 9.6 |
| Pixel 10 parity, short/medium/near-capacity prompts | Pending |
| End-to-end speech completion, two voices, long streaming chunks | Pending |
| Paired CPU-int8 timing, energy, first audio, thermal/RSS review | Pending |
| Real fp16/int8 KV graph I/O and quality assessment | Not implemented; harness simulation leaves float32 staging intact; fp16/int8 byte counts are analytical equivalents, not measured traffic |

No capacity or KV-precision variant is a performance win until it passes the
research brief's paired CPU-int8 protocol with the graph checksum, runtime
versions, device fingerprint, workload, power method, and listening review
recorded. The precision path needs a real lower-precision graph input and
supported Android tensor-buffer writes before transfer savings can be claimed.
