# Using the Pocket TTS library

Three Gradle modules:

| module | what |
|---|---|
| `:pockettts-core` | the engine: LiteRT graphs, host state, tokenizer, model delivery. `dev.pockettts` |
| `:pockettts-service` | `TextToSpeechService` (`dev.pockettts.service.PocketTtsService`), registered by its own manifest |
| `:app` | demo UI; depends on both and nothing else of the engine |

```kotlin
dependencies {
    implementation(project(":pockettts-core"))     // engine only
    implementation(project(":pockettts-service"))  // + system TTS engine
}
```

## Models: where the files come from

`CompiledModel.create` needs a real path, so every source materializes a file.
Sources are chained; the first one that has a file wins.

```kotlin
// Default: adb-pushed / app-owned directory first, GitHub release fallback.
val models = PocketTtsModels.default(context)          // version "2026.09"

// A release only (cold install with no pushed files).
val release = ReleaseModelSource(
    releaseBase = "https://github.com/geneing/PocketTTS-LiteRT/releases/download/models-2026.09",
    cacheDir = File(context.noBackupFilesDir, "pockettts/2026.09"),
)
val models = PocketTtsModels.ofRelease(release, DirectorySource(devDir))
```

`ReleaseModelSource.ensure(names)` fetches `models.json`, downloads the variants
those files live in to `<name>.part` (resuming via `Range` when interrupted),
verifies SHA-256, and extracts the stored zip entries atomically. `PocketTtsEngine`
can do it for you:

```kotlin
val required = PocketTtsEngine.requiredFiles(config)   // what this placement needs
engine.ensureModels { done, total -> /* progress */ }  // no-op when already present
```

Variants keep downloads small: `base` (~81 MB: assets, tokenizer, voices, dec_tx,
SEANet windows) plus one LM pack (~86 MB int8). A 512 MB LM/decoder set is never
downloaded for a device that runs int8. `HttpDownloader` is the default transport;
implement `Downloader` to plug in OkHttp.

## Engine and session

One `PocketTtsEngine` per process — it owns the graphs, the mapped host assets, the
K/V + decoder scratch buffers, a small LRU of repacked voice states, and the
worker thread. `newSession(voice)` is cheap: it adds only per-utterance position
and RNG, not ~25 MB of buffers or a re-read of the voice file, so an app can keep
one engine for its lifetime and take a session per request.

```kotlin
val engine = PocketTtsEngine(context)                     // seconds, ~150 MB native
val session = engine.newSession("alba")
```

`PocketTtsConfig` overrides the policy: `placement`, `lmGraph`, `lmSteps`,
`streamW`, `noiseSeed` (deterministic repeats), `gpuCache` (serialized GPU program
cache). `Placement.default` picks the split per device: flow-LM int8 on CPU, the
Mimi decoder transformer on the Tensor G5 NPU when the `_g5` graph and the dispatch
shim are both installed, SEANet streaming on the GPU. `force_cpu.txt`,
`force_gpu.txt` and `force_fp32.txt` in the external files dir still override.

### Three entry points

```kotlin
// 1. blocking, whole utterance
val r: TtsResult = session.synthesize("Hello there.")

// 2. blocking, but audio as it decodes
engine.stream("Hello there.", "alba") { chunk -> track.write(chunk, 0, chunk.size, BLOCKING) }

// 3. asynchronous + cancellable (barge-in)
val u: Utterance = session.speak("Hello there.", listener)
u.cancel()
```

`TtsProfile` carries the timings: `firstChunkMs` (time to first audio),
`audioChunks`, and the per-stage `lmInMs`/`lmRunMs`/`lmReadMs`/`decTxMs`/`seanetMs`.

`firstChunkMs` includes engine construction in practice (the graphs compile before
the first utterance), so measure it on a warm engine. On a Pixel 10 the flow-LM is
the floor at ~0.86 s, and first audio lands at ~1.2 s once the engine is loaded.

### Speech rate and pitch

`session.rate` (tempo, 1.0 = natural) and `session.pitch` (tone, 1.0 = natural) are
independent post-processing on the decoded 24 kHz PCM, applied after the graphs so
the LM/Mimi graphs and the RNG stream never see a different rate. Rate changes
tempo with pitch unchanged; pitch changes tone with the duration unchanged; the two
compose, and `1f, 1f` is an exact pass-through. Both are captured when a call
starts (so changing them mid-utterance applies to the next call) and are clamped to
0.5x–2x.

```kotlin
session.rate = 0.8f    // slower, same voice
session.pitch = 1.2f   // higher, same duration
```

The DSP is the vendored [Sonic](https://github.com/waywardgeek/sonic) library
(`pockettts-core/src/main/java/sonic/Sonic.java`, Apache-2.0, © Bill Cox), used as a
constant-pitch time-stretch plus a resample. It is 16-bit internally, so anything
other than the identity path is requantized.

### Voice agent: push text as the LLM emits it

`begin()` returns a push interface: feed fragments, and each complete sentence is
synthesized immediately rather than waiting for the reply to finish.

```kotlin
val speech = session.begin(listener)
speech.push("Sure — here is the ")
speech.push("answer you asked for.")
speech.end()          // flush the tail and deliver onDone
speech.cancel()       // barge-in: stop generating and discard queued audio
```

Generation happens on the calling thread, so drive it from your own worker and
write chunks to a pre-opened `AudioTrack` (use `USAGE_ASSISTANT` with
`PERFORMANCE_MODE_LOW_LATENCY`). The `session.speak`/`begin` calls run on the
engine's internal single worker, which serializes graph invocations — a second
utterance starts only after the first yields or is cancelled.

## System TTS engine

`:pockettts-service` declares the service, so adding the dependency registers
"Pocket TTS" in Android's TTS settings. Nothing further is required in the app
except provisioning the models (bundled assets, adb push, or `ensureModels`).

It implements `onSynthesizeText` by streaming PCM-16 chunks, so playback starts at
the first SEANet window rather than after the utterance, and `onStop` maps to
session cancellation. It advertises `eng` only; the voices are exposed through
`PocketTts.VOICES` and passed as the request's voice name. `request.speechRate` and
`request.pitch` are honoured independently; when a client sends the framework
default (100, "no preference"), the engine's persisted rate/pitch from the settings
screen apply instead.

## Development loop

The directory source is what makes iterating on models fast: `adb push` a rebuilt
graph over the existing one and relaunch — no APK rebuild, no asset-pack
`install-multiple`, no immutable pack contents.

```bash
# publish the packs (stored zips + a filled models.json) and upload them
python3 scripts/pack_models.py --out scripts/out --dist scripts/dist
gh release create models-2026.09 --title "Models 2026.09" \
    scripts/dist/pockettts-*.zip scripts/dist/models.json

# test the download path end to end against a local server
python3 scripts/serve_models.py --port 8099 &
adb reverse tcp:8099 tcp:8099
```

`models.json` (committed) is the version contract between library and weights: it
maps each file to the variant that carries it, with a SHA-256 per zip.
