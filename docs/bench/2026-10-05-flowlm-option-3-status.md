# FlowLM option 3 implementation status

Date: 2026-10-05. Device: Pixel 10 / Tensor G5. Focused harness checks,
capacity safety probes, the named-buffer regression check, and short speech
pairs on two voices were run on-device; medium/long and power acceptance remain
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
| Pixel 10 parity, short/medium/near-capacity prompts | Short prompt matched 512 output bytes exactly; 46-token prompt was safely rejected at 256 and completed using 512; 256/512 named shape/write check passed |
| End-to-end speech completion | Short streaming pair passed on Alba and Marius in both orders; medium/long speech and long streaming chunks pending |
| Paired CPU-int8 timing, energy, first audio, thermal/RSS review | Short speech timing measured; no energy comparison; medium/long, sustained thermal, and listening review pending |
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
cannot safely produce full speech in the 256 bucket. Four short streaming
speech A/B runs are now recorded below; they do not include power measurements
or medium/long speech.

The first full `shortSpeechCapacityPair` attempt failed before audio with a
LiteRT host buffer error (`2052` bytes available, `32832` bytes supplied)
while writing the step mask. The graph files advertised the expected named
step mask and KV shapes, but `PocketTtsEngine` selected buffers positionally.
The engine now binds and runs `serving_default` by name and checks mask/KV
dimensions against `lmCapacity` before synthesis. After rebuilding and
installing the option 3 APKs, `capacityGraphNamedStepShapesMatchHostBuffers`
passed on Pixel in 4.227 s and exercised the 256- and 512-position mask writes.
The failure was a real app buffer-binding bug and is covered by the regression
test; it was not a capacity-planning rejection.

Host inspection of the actual artifacts found `serving_default` at signature
index 1 in both files, with mask shapes `[1,16,1,257]` and `[1,16,1,513]` and
K/V shapes `[1,96,256,64]` and `[1,96,512,64]`. The Android test compile and
on-device named-binding shape/write regression check passed.

The corrected full short-speech pair passed in both run orders for Alba and
Marius. Each pair used the same text (`Hello there, how are you?`), seed 42,
streaming mode, and production decoder placement (LM CPU, decoder transformer
NPU, SEANet GPU). The candidate graph SHA-256 was
`a9a7147c06f7f474346805b66a756b64d19f5e0eb49468c253eb78f12e060b8a`; the
512-position control was
`895cd59e4c8256000e9bd3855e6b6f87f00e3ad9a2b093cbe70df018370a10f9`.

| Voice | 256 candidate synthesis / first audio | 512 control synthesis / first audio | Output | Two-run median synthesis |
|---|---:|---:|---|---:|
| Alba | 637 / 637 ms, 605 / 605 ms | 771 / 771 ms, 919 / 919 ms | 18 frames / 1.44 s for both; waveform corr 0.999999999999998 | 621 vs 845 ms; 1.36x faster |
| Marius | 626 / 626 ms, 544 / 544 ms | 578 / 578 ms, 825 / 825 ms | 11 frames / 0.88 s for both; waveform corr 0.999999999999987 | 585 vs 701.5 ms; 1.20x faster |

Each comma-separated measurement is one order: candidate-first, then
reference-first. These are two runs per graph, not a p50/p95 distribution.
Candidate engine-load times varied from 2.27 to 3.21 s; control loads varied
from 2.26 to 2.89 s, so the short synthesis gain does not establish a cold
start improvement. No speech was truncated in these four runs. Listening,
transcription, energy, and sustained thermal review have not been performed.
Reports and WAVs are retained under
`scripts/out/option3-speech-device/speech-*/` in this worktree.

## Reproduction commands

Build the 256 variant with the pinned WSL reference checkout and conversion
environment, then push `pt_flowlm_fused_dyn8_all_pmax256.tflite` beside the
shipped 512 graph. Run one instrumentation method at a time:

```powershell
.\gradlew.bat :app:installDebug :app:installDebugAndroidTest
adb push build\option3-graphs\256\pt_flowlm_fused_dyn8_all_pmax256.tflite /sdcard/Android/data/com.pockettts/files/
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#capacityPlannerAndVoicePrefixRepackingAreExact com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#capacityGraphNamedStepShapesMatchHostBuffers -e lmCapacity 256 com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#runTextPromptHarness -e lmGraph pt_flowlm_fused_dyn8_all.tflite -e lmCapacity 256 -e backends CPU -e voice alba -e text 'Hello.' -e kvPrecision fp32 com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -e class com.pockettts.FlowLmHarnessTest#shortSpeechCapacityPair -e lmCapacity 256 -e voice alba -e text 'Hello there, how are you?' -e mode stream -e order candidate_first com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb pull /sdcard/Android/data/com.pockettts/files/flowlm-harness
```

The short-speech command above was completed for both Alba and Marius in both
orders. Remaining acceptance is medium/long speech when the full planned frame
budget fits the chosen bucket, one-shot comparison, paired Android power
monitor runs with audio-only subtraction, RSS/temperature checks, and listening
review. The 256 bucket cannot cover the 46-token Mary case with its 217-frame
budget, so that case must use 512 or a separately proven fallback. Near-capacity
prompt parity remains a prompt-cache diagnostic, not a speech test. No
production capacity policy change is claimed from the short pairs alone.
