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
  transfer bytes, graph SHA-256, device fingerprint, per-step EOS/latent/new-K/
  new-V differences, and a deterministic zero-noise first decode step. Its
  `fp16` and `int8_per_head` cache modes are host-side
  round-trip simulations only: LiteRT graph inputs still use float32, so they
  do not reduce actual transfer bytes or represent a speed result.
- The bundled voice files have 126 prefix frames (Alba, Charles, Javert,
  Marius, Mary) or 133 (Eve), calculated from their compact fp16 file sizes.
  Even the minimum planned audio budget makes 128 too small for full speech
  with these voices. The exporter uses an empty prefix for a parity fixture
  that would cross the bucket boundary and rejects a fixture wider than PMAX.

For a Linux/WSL export, create an isolated `PT_OUT` directory and run the
`fused` stage followed by `quant` for each capacity. For example, capacity 256
produces `pt_flowlm_fused_fp16_pmax256.tflite` and
`pt_flowlm_fused_dyn8_all_pmax256.tflite`. Capacity 128 is exportable for a
custom short-prefix voice. Alba's 126-frame prefix fits the static shape, but
126+16 exceeds the exporter's prefill parity fixture, so that fixture uses an
empty prefix. The app rejects an undersized bucket before prefill.

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
| Exporter syntax | Passed with bundled Python AST parse |
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

The documented reference clone and conversion virtual environment were absent
at the repository paths in WSL, so no reduced-capacity `.tflite` was generated
in this checkpoint. The 128/256/512 graph parity and Pixel 10 rows are pending.

## Exact device follow-up

Build the 256 variant with the pinned WSL reference checkout and conversion
environment, then push `pt_flowlm_fused_dyn8_all_pmax256.tflite` beside the
shipped 512 graph. Run one instrumentation method at a time:

```powershell
.\gradlew.bat :app:installDebug :app:installDebugAndroidTest
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#capacityPlannerAndVoicePrefixRepackingAreExact com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#runTextPromptHarness -e lmGraph pt_flowlm_fused_dyn8_all.tflite -e lmCapacity 256 -e backends CPU -e voice alba -e text 'Hello.' -e kvPrecision fp32 com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb pull /sdcard/Android/data/com.pockettts/files/flowlm-harness
```

Repeat with a second bundled voice, 1-5 and 25-50 token prompts, and a
near-capacity prompt that still fits the planned frames. Use matched 512 runs
and reverse A/B order. Inspect every prompt row and the first decode row;
`first_step_over_1e-3` is diagnostic, not an acceptance gate. Full short,
medium, and long speech, one-shot and streaming, paired Android power monitors
with audio-only subtraction, RSS, temperatures, listening, and the graph,
runtime, dispatch, and AOT details still need separate acceptance runs under
the research protocol.
