# FlowLM option 3 implementation status

Date: 2026-10-05. Device: Pixel 10 / Tensor G5. Focused harness checks and
capacity safety probes were run on-device; full speech acceptance remains
pending.

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
- `FlowLmHarnessTest.shortSpeechCapacityPair` runs a complete short utterance
  through the 256 candidate and 512 production control, one engine at a time
  with the same seed and decoder placement. It saves both WAVs and reports
  synthesis, first audio, graph load, LM, Mimi, frames, and waveform correlation.
  The method accepts `text`, `voice`, `lmCapacity`, `mode`, and `order` arguments.
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
| 256 graph export and tensor parity | Passed WSL export, one-step eager parity, and focused device comparison |
| 128 graph export | Pending; no bundled voice fits its planned audio budget |
| Exact prefix repack / bucket planner | Passed on Pixel 10 |
| Android Kotlin compile | Passed: `:app:compileDebugAndroidTestKotlin` with Gradle 9.6 |
| Pixel 10 parity, short/medium/near-capacity prompts | Short prompt matched 512 output bytes exactly; 46-token prompt was safely rejected at 256 and completed using 512 fallback; near-capacity speech not run |
| End-to-end speech completion, two voices, long streaming chunks | Pending |
| Paired CPU-int8 timing, energy, first audio, thermal/RSS review | Not run; short one-step timing only |
| Real fp16/int8 KV graph I/O and quality assessment | Not implemented; harness simulation leaves float32 staging intact; fp16/int8 byte counts are analytical equivalents, not measured traffic |

No capacity or KV-precision variant is a performance win until it passes the
research brief's paired CPU-int8 protocol with the graph checksum, runtime
versions, device fingerprint, workload, power method, and listening review
recorded. The precision path needs a real lower-precision graph input and
supported Android tensor-buffer writes before transfer savings can be claimed.

The pinned reference clone (`001cf6e`) was placed in the ignored
`references/pocket-tts` directory, and the 256 graph was exported with the
existing WSL conversion environment. Its first conversion finished, but the
parity check revealed that the exporter used signature 0 (prefill) for the
fused one-step check; the correct step signature is 1. The exporter now selects
the right index and supports `PT_REUSE_FUSED=1` to repeat parity on the exact
existing artifacts without paying for conversion again.

| 256 artifact | Bytes | SHA-256 |
|---|---:|---|
| `pt_flowlm_fused_dyn8_all_pmax256.tflite` | 87,504,288 | `a9a7147c06f7f474346805b66a756b64d19f5e0eb49468c253eb78f12e060b8a` |

The 256 fp16 one-step output matched eager with correlation 1.000000 and
max absolute difference `1.02e-5`. Its prefill signature's new K/V differed
from sequential eager by at most `1.19e-5`/`3.81e-6`. The dynamic-int8 graph
versus eager reported latent/K/V correlations `0.999813`/`0.999912`/`0.999164`
and EOS max absolute difference `0.656`; this is a quantized one-step probe,
not a speech-quality result. Logs are in the ignored `build/option3-fused-256-parity.log`
and `build/option3-quant-256.log` in this worktree.

## Pixel 10 focused device results

The 256 graph was pushed to the Pixel 10 and compared with the shipped 512
dynamic-int8 graph using `FlowLmHarnessTest.runTextPromptHarness`, CPU backend,
voice alba, text `Hello`, and capacity-aware planning. Both selected `Hello`
candidate outputs had SHA-256
`b482c427d5908c1c4a2e5defe2f061166bb6dd637226a148907049b89542ca2f`.
The measured 256 run loaded in 1,750.73 ms, staged inputs in 6.42 ms, ran in
19.72 ms, and read outputs in 0.24 ms; the 512 control loaded in 1,736.06 ms,
staged in 13.44 ms, ran in 26.51 ms, and read in 0.32 ms. KV input staging
bytes were 12,582,912 vs 25,165,824. This is one short prompt run per graph,
not a median or an end-to-end speech comparison. The 256 run's correlation to
the CPU fp16 reference was `0.99992840`, mean absolute difference `0.006461578`,
and mean square difference `0.000109081`.

For 46 Mary prompt tokens, the planner estimated 217 generated frames. The 256
candidate failed before inference with the explicit capacity error
`voice=126 + prompt=46 + plannedFrames=217`; recommended fallback was 512.
The 512 candidate then completed: load 1,756.61 ms, stage 69.91 ms, run
542.39 ms, read 5.02 ms, and KV input staging 25,165,824 bytes. Its correlation
to CPU fp16 was `0.99993567`, mean absolute difference `0.006524455`, and mean
square difference `0.000119100`. This matches the expected capacity policy;
it does not establish a 256-token speed win for medium or long speech.

The 360-token near-capacity control remains a prompt-cache diagnostic and
cannot safely produce full speech in the 256 bucket. The focused device checks
did not capture speech audio or power measurements.

## Exact device follow-up

Build the 256 variant with the pinned WSL reference checkout and conversion
environment, then push `pt_flowlm_fused_dyn8_all_pmax256.tflite` beside the
shipped 512 graph. Run one instrumentation method at a time:

```powershell
.\gradlew.bat :app:installDebug :app:installDebugAndroidTest
adb push build\option3-graphs\256\pt_flowlm_fused_dyn8_all_pmax256.tflite /sdcard/Android/data/com.pockettts/files/
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#capacityPlannerAndVoicePrefixRepackingAreExact com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#runTextPromptHarness -e lmGraph pt_flowlm_fused_dyn8_all.tflite -e lmCapacity 256 -e backends CPU -e voice alba -e text 'Hello.' -e kvPrecision fp32 com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#shortSpeechCapacityPair -e lmCapacity 256 -e voice alba -e text 'Hello there, how are you?' -e mode stream -e order candidate_first com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
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
