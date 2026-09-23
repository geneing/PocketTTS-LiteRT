# AGENTS.md — converting Pocket TTS to LiteRT `.tflite`

`scripts/build_pockettts.py` re-authors Kyutai's **Pocket TTS** (100M flow-matching
LM over 32-dim Mimi latents + a tiny Mimi codec) into fixed-shape, stateless
LiteRT `CompiledModel` graphs plus the host-side assets the Android app
(`com.pockettts`) loads. It also runs the eager PyTorch reference and prints a
parity number for every graph, so "did the conversion work" is answered by the
script itself.

The graphs and the re-authoring tricks are described in the module README; this
file is only about running the export.

## TL;DR

```bash
# 1. reference implementation at the pinned commit (gitignored under references/)
git clone https://github.com/kyutai-labs/pocket-tts.git references/pocket-tts
git -C references/pocket-tts checkout 001cf6e

# 2. uv environment (Linux/WSL2; see "Platform" below)
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
printf 'torch==2.12.1+cpu\n' > constraints.txt
uv pip install -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    -e references/pocket-tts -r scripts/requirements-convert.txt

# 3. convert everything (writes scripts/out/, gitignored)
PYTHONPATH=$(pwd)/references/pocket-tts PT_OUT=$(pwd)/scripts/out \
    python scripts/build_pockettts.py all
```

No manual model download and no Hugging Face token are needed — the script pulls
what it needs (see "What it downloads").

## Platform: Linux (or WSL2 on Windows)

`litert-torch` depends on `litert-converter`, which ships **only macOS-arm64 and
manylinux wheels** — there is no Windows wheel, so the export cannot run on
native Windows. On a Windows host run the whole thing in WSL2 (Ubuntu); the repo
on `/mnt/<drive>/...` is fine to read, and the reference clone can live on the
WSL side. This conversion was verified in WSL2 Ubuntu (x86_64).

## Environment

Verified versions (all CPU):

| tool | version |
|---|---|
| Python | 3.10.20 (uv-managed) |
| uv | 0.11.6 |
| torch | 2.12.1+cpu |
| litert-torch | 0.9.3 |
| ai-edge-litert | 2.1.6 |
| ai-edge-quantizer | 0.8.0 |
| litert-converter | 0.3.1 |
| jax | 0.6.2 |
| pocket-tts | 3.0.2 (clone `001cf6e`) |

Pins live in `scripts/requirements-convert.txt`; `pocket-tts` is installed from
the clone (`-e`), which brings its own runtime deps. Install CPU torch first from
`https://download.pytorch.org/whl/cpu` and pass the `torch==2.12.1+cpu` constraint
on the second `uv pip install`, otherwise uv resolves the ~3 GB CUDA wheel from
PyPI. The `litert-torch==0.9.3` / `ai-edge-litert==2.1.6` / `ai-edge-quantizer==0.8.0`
triple matters: `to_fp16` uses the 0.8.x `ai_edge_quantizer` API, and 0.9.x
`litert-torch` requires it. `litert-converter` is unpinned (`0.3.*` via
litert-torch) and is the one dependency that drifts.

## What it downloads

Cached under `~/.cache/huggingface/hub`; nothing is committed.

- **LM + Mimi weights** — `TTSModel.load_model()` with the default English config
  tries the gated `kyutai/pocket-tts` (`…/english/model.safetensors@39592ff2…`)
  and **falls back to the ungated** `kyutai/pocket-tts-without-voice-cloning`
  (`…@d29db79…`) when that download is refused. No token is required for the
  fallback; the two bundles share the LM/Mimi weights.
- **Voice states** — always `kyutai/pocket-tts-without-voice-cloning`
  (`languages/english/embeddings/{name}.safetensors@e81d79e…`).
- **Tokenizer** — `kyutai/pocket-tts-without-voice-cloning`
  (`languages/english/tokenizer.model@d29db79…`).

To force the ungated weights even when the gated repo is cached, set `HF_TOKEN`
to an invalid value or clear `~/.cache/huggingface/hub/models--kyutai--pocket-tts`.

## Outputs (`PT_OUT`, default `scripts/out/`)

| file | what |
|---|---|
| `pt_flowlm_step{,_fp16}.tflite` | one AR step, packed-KV I/O (reference; not loaded by the app) |
| `pt_flow_head{,_fp16}.tflite` | flow head alone (reference; not loaded by the app) |
| `pt_flowlm_fused.tflite` | step + head fused, fp32 reference |
| `pt_flowlm_fused_dyn8_all.tflite` | the same graph dynamic-range int8 — **the app's LM**; also carries a head-less batched `prefill` signature sharing the backbone's weight buffers, but the app does **not** use it: on this graph the batched rows do not reproduce the per-token step (uncorrelated audio, short prompts truncated), so `usePrefill` is off and each prompt is one fused step per token |
| `pt_flowlm_fused_fp16.tflite` | fp16 sibling of the fp32 graph, kept for quantizer comparison only — not packed, not loaded |
| `pt_flowlm_prefill{N}{,_fp16}.tflite` | standalone head-less batched prefill (`build_pockettts.py prefill`, `PT_PREFILL_TOKENS`) — a diagnostic for the prefill math; not packed, and nothing loads it (the app prefills per token) |
| `pt_mimi_dec_tx{,_fp16}.tflite` | Mimi decoder transformer block (app runs CPU) |
| `pt_mimi_deconly{,_fp16}.tflite` | SEANet decoder |
| `pt_embed_f16.bin`, `pt_input_linear_f32.bin`, `pt_bos_input_f32.bin`, `pt_neutral_latent_f32.bin` | host assets |
| `pt_tokenizer.tsv` | sentencepiece pieces for the Kotlin unigram encoder |
| `pt_voice_{alba,marius,javert,charles,mary,eve}.bin` | preset voice KV caches (CC-BY-4.0 / CC0 only) |
| `pipeline_{tflite,eager}.wav` | parity artifacts, not shipped |

`_fp16` files are weight-only fp16 (fp32 compute) via `ai-edge-quantizer`'s
`FLOAT_CASTING` recipe; `_dyn8_all` is dynamic-range int8 (int8 weights, fp32
activations, so the host protocol is unchanged) from the same tool. The Android
app loads the int8 LM plus the fp16 Mimi graphs, the four host `.bin` assets, the
tokenizer and the voices, and prefills each prompt one fused step per token
(`usePrefill = false`); the fp16 flow-LM is a build byproduct, not a fallback.

## Verifying success

The script prints these on `all`; every line should match. Absolute float
differences are expected from the fitted tanh-polynomial erf-GELU — it is the
only non-exact rewrite — and are largest on the flow-LM latents (they compound
autoregressively). Correlation is the acceptance signal.

| check | result |
|---|---|
| flow-LM step vs eager (teacher-forced, 41 steps) | latent max\|d\| ~1.4e-2, cond corr 1.000000 |
| flow head vs eager `lsd_decode` | max\|d\| ~2e-7 |
| fused vs split step+head (12 free-run steps) | max\|d\| 0.0 |
| fused `prefill` signature vs per-token step | **not verified.** This row read ~1.2e-5 until the check was fixed: it passed signature index 1 (the step) with the prefill's *inputs*, so it measured nothing. Now index 0; expect it to report the mismatch that keeps `usePrefill` off |
| dec_tx blocks vs full-sequence eager | corr 1.000000, max\|d\| ~5e-4 |
| deconly + dec_tx vs eager decode | corr 1.000000, max\|d\| ~2e-4 |
| full tflite pipeline vs eager (same noise) | audio corr ~0.997 |
| tokenizer pieces | 4000 |

Stages run individually: `flowlm | head | fused | prefill | dectx | deconly | assets | pipeline`,
plus `quant | multistep | stream` for the optional variants.
`PT_OUT` must be a POSIX path (do not point it at a Windows-style `C:\...` path
from inside WSL — it becomes a literal relative directory named `C:...`).

### Byte-identity is not guaranteed across machines

Re-running here produced the same file *sizes* and passed every parity check,
but only **16 of 21** files matched the published HF repo
(`mlboydaisuke/Pocket-TTS-LiteRT`) by sha256; the fused/head graphs and
`pt_bos_input_f32.bin` differed by ~1 ULP (e.g. bos max\|d\| 4.7e-10, corr
1.000000000). That is host BLAS/oneDNN rounding in the baked constants, not a
logic change. Treat the parity numbers, not the checksums, as the contract.

## Handoff to the Android app

```bash
./scripts/install_to_device.sh                 # default src is scripts/out/ (or pass a dir)
./gradlew :app:installDebug
```

The engine, model delivery and TTS service live in the library modules
(`:pockettts-core`, `:pockettts-service`); `:app` is a thin demo on top. See
`docs/library.md`. `install_to_device.sh` pushes the files the engine actually
loads (`PocketTtsEngine` constants in `:pockettts-core`) into
`/sdcard/Android/data/com.pockettts/files/`, which `PocketTtsModels.default`
resolves first — that directory source is what makes model iteration a plain
`adb push` with no APK rebuild.

Beware: `:app:connectedDebugAndroidTest` **uninstalls the app** when it finishes,
which deletes `/sdcard/Android/data/com.pockettts/files/` and every pushed model.
For on-device test runs install first and drive the runner directly:

```bash
./gradlew :app:installDebug :app:installDebugAndroidTest
adb shell am instrument -w -e class com.pockettts.LatencyProbeTest \
    com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
```

`LatencyProbeTest` covers the engine (`probe` one-shot-vs-streamed parity,
`continuity`, and `prefillPaths`, which logs the batched-vs-per-token prefill
gap); `TtsServiceInstrumentedTest` drives the real framework client.

To distribute models instead, `scripts/pack_models.py` builds the stored zips and
fills `models.json`; upload them as a `models-<version>` GitHub release, which
`ReleaseModelSource` then downloads on demand (variants: `base`, `lm-int8`,
`npu-g5`; `lm-quant-variants` and `lm-multistep` are declared in `models.json` but
still `PENDING`, so the packer exits 1 with warnings).
`scripts/serve_models.py` + `adb reverse` tests that path locally.

Speech rate and pitch are host-side DSP over the decoded PCM, not a graph change.
The time-stretch/pitch-shift is the vendored [Sonic](https://github.com/waywardgeek/sonic)
library (`pockettts-core/src/main/java/sonic/Sonic.java`, Apache-2.0, © Bill Cox),
wrapped by `SonicStretcher` in `:pockettts-core`.

The app can benchmark placements and compare audio quality against the
CPU/CPU/CPU gold — see the "Benchmark" button (or
`adb shell am start -n com.pockettts/.MainActivity --ez bench true`).
