# Pocket TTS on LiteRT CompiledModel (full-GPU)

[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (Kyutai, ~100M params) running
on-device with LiteRT `CompiledModel`. Pocket TTS is a **flow-matching LM over continuous
32-dim Mimi latents**: per 12.5 Hz frame a 6-layer/1024-wide causal transformer conditions a
6-block AdaLN MLP flow head that turns one Gaussian draw into the next latent (Lagrangian
Self Distillation, 1 step — no diffusion loop); a 20M tiny Mimi (×16 ConvTranspose upsample +
2-layer transformer + SEANet) decodes latents to 24 kHz audio. No FFT anywhere in the
pipeline, which is exactly why this architecture fits the GPU delegate.

This is the first flow-matching LM in this zoo, and the first TTS here whose **entire
pipeline — language model, flow head and codec decoder — runs on the GPU**.

| | |
|---|---|
| Model | [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) (CC-BY-4.0); ungated weights: [pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) |
| Converted | [mlboydaisuke/Pocket-TTS-LiteRT](https://huggingface.co/mlboydaisuke/Pocket-TTS-LiteRT) |
| Backbone | 6-layer transformer, d=1024, 16 heads, GELU FFN ×4, LayerNorm, RoPE |
| Flow head | SimpleMLPAdaLN, 6 res blocks × 512, 2 time conditions (LSD) |
| Codec | pocket Mimi: quantizer-free 32-dim latents, ×16 upsample, 2-layer transformer (context 250), SEANet (ratios 6·5·4) |
| Rates | 12.5 Hz latent frames · 1920 samples/frame · 24 kHz mono |
| Voices | precomputed prompt-state KV caches (alba, marius, javert, charles, mary, eve — CC-BY-4.0/CC0 only) |
| App | `com.pockettts` — pick a voice, type text, generate, play |

## Best configuration (Pixel 10, Tensor G5)

Measured on a Pixel 10: **2.16x real-time** — 5.60 s of speech in 2.56 s — against
0.84x for the all-CPU reference. One command reproduces it:

```bash
cd pockettts/
scripts/reproduce_best.sh            # writes the weights into scripts/out/
scripts/reproduce_best.sh push apk   # and gets them onto a phone
```

| stage | graph | where | why |
|---|---|---|---|
| flow-LM (fused step + flow head) | `pt_flowlm_fused_dyn8_all.tflite` | **CPU**, dynamic-range int8 | The step is DRAM-bandwidth-bound, not FLOP-bound: 84.44M weights stream from DRAM every frame while the activations are batch-1. int8 weights cut that traffic 4x, the step 2.27x (19.6 → 8.65 ms/frame), and the file to 82 MB from 161 MB. |
| Mimi decoder transformer | `pt_mimi_dec_tx_fp16_g5.tflite` | **NPU**, AOT-compiled for Tensor G5 | 131 ms vs 489 ms on CPU and 142 ms on GPU, and unlike the GPU delegate it is faithful to the CPU reference (corr 0.9999). |
| SEANet decoder | `pt_mimi_deconly_fp16.tflite` | **GPU** | 1.36 s vs 4.6 s on CPU; the GPU output is clean on this stage. |

The app picks this automatically when the files are present — `PocketTtsSynthesizer`
prefers the int8 flow-LM, `Placement.default` opts into the NPU decoder
transformer when both the `_g5` graph and the dispatch shim are installed, and
SEANet defaults to the GPU. `force_cpu.txt` / `force_gpu.txt` / `force_fp32.txt`
still override. Decisions and rejected variants: `docs/int8_lm.md`; the joint
CPU/GPU/NPU timing tables: `docs/RESULTS.md`.

### Steps to reproduce the weights

`scripts/reproduce_best.sh` runs all of these in order; each stage also runs alone
(`build`, `quant`, `aot`, `shim`, `push`, `apk`, `env`, `summary`). All of it must
run under **Linux or WSL2** — `litert-converter` has no Windows wheel.

1. **Reference clone** at the pinned commit:
   ```bash
   git clone https://github.com/kyutai-labs/pocket-tts.git references/pocket-tts
   git -C references/pocket-tts checkout 001cf6e
   ```
2. **Conversion venv** (Python 3.10, CPU torch, pinned stack) — see `AGENTS.md` and
   `scripts/requirements-convert.txt`. Defaults to `~/pockettts-conv/.venv`.
3. **Base graphs + host assets** — `python scripts/build_pockettts.py all`. Downloads
   the ungated `kyutai/pocket-tts-without-voice-cloning` weights, runs ~3 minutes, and
   writes the fp16 graphs, the host `.bin` assets, the tokenizer and the voices into
   `scripts/out/`.
4. **int8 flow-LM** — `PT_QUANT=dyn8_all python scripts/build_pockettts.py quant`.
   Dynamic-range int8, channelwise weights, fp32 activations, **no calibration**; the
   graph's inputs and output stay fp32, so nothing in the host protocol changes.
5. **AOT-compile the decoder transformer** —
   `python scripts/aot_tensor_g5.py pt_mimi_dec_tx_fp16`, which needs its own venv
   (`ai-edge-litert==2.2.0` + `ai-edge-litert-sdk-google-tensor==2.2.0`, default
   `~/pockettts-aot/.venv`) and writes `pt_mimi_dec_tx_fp16_g5.tflite`. Optional:
   without it `dec_tx` runs on CPU and only that stage slows down.
6. **Dispatch shim** — `scripts/fetch_google_tensor_dispatch.sh` pulls
   `libLiteRtDispatch_GoogleTensor.so` from the matching LiteRT release (v2.2.0) into
   `app/src/main/jniLibs/arm64-v8a/`. The shim, the Android `litert` runtime and the
   AOT compiler must all be the same LiteRT release, or the NPU silently never engages.
7. **Device** — `scripts/install_to_device.sh`, then `./gradlew :app:installDebug`.
   `ADB=<path>` overrides the adb binary, which is what you need on Windows: the WSL
   adb server cannot see the device.

The int8 flow-LM and the `_g5` AOT graph are produced by this branch and are **not**
in the published [mlboydaisuke/Pocket-TTS-LiteRT](https://huggingface.co/mlboydaisuke/Pocket-TTS-LiteRT)
bundle, which carries the fp16 graphs.

## Graphs and placement

Every graph is stateless; KV caches, RoPE tables, the token-embedding lookup, the
32→1024 latent projection, noise draws and EOS logic live in Kotlin
(`PocketTtsSynthesizer.kt`) — the dia2/vibevoice packed-KV pattern.

| graph | I/O | fp16 size |
|---|---|---|
| `pt_flowlm_fused` | emb[1,1,1024] + cos/sin[1,1,1,64] + mask[1,16,1,513] + pk/pv[1,96,512,64] + noise[1,32] → [1,12321] = eos ∣ latent ∣ new-k ∣ new-v | 169 MB |
| `pt_flowlm_fused_dyn8_all` | the same I/O (int8 weights only, activations still fp32) | **82 MB** |
| `pt_mimi_dec_tx` | lat[1,65,32] → feat[1,512,1024] | 17 MB |
| `pt_mimi_deconly` | feat[1,512,4096] → audio[1,1,491520] | 11 MB |
| `pt_flowlm_step` / `pt_flow_head` | the same frame split into two graphs (cond exposed) — reference variant, not loaded by the app | 151 + 18 MB |

The app runs the **fused** frame graph: on Mali the per-frame cost is dispatch/sync-bound,
not FLOP-bound — with the split graphs one frame measured ~11 ms input write + ~12 ms
`run()` + ~20 ms readback (the readback hides the asynchronous GPU completion, spread over
four separate output reads) plus a second invocation for the head. Fusing the head and
concatenating everything into one output removes one invocation and three readbacks per
frame (−1.1 s on an 8 s utterance, measured).

**Placement: the Mimi decoder transformer (`pt_mimi_dec_tx`) ships on CPU.** Every graph
compiles fully on the GPU (`Replacing N out of N node(s) with delegate (LITERT_CL)`) on
both devices tried, but on the Pixel 8a's Mali the GPU OUTPUT of that one graph is audibly
degraded — gravelly, hissy voicing. Measured on the alba voice: HNR 0.9 dB and −32 dB
high-band noise on GPU vs **2.8 dB / −37 dB on CPU, which matches the fp32 desktop eager
reference exactly**; requesting `GpuOptions(precision = FP32)` does NOT recover it (0.6 dB),
so this is not fp16 rounding — the same decoder-transformer behavior the mimi/ module
documents, and the σ-VAE class of finding from vibevoice/. The SEANet graph on GPU is
clean (CPU-pinning it changes nothing but speed), as are the LM and flow head. The
transformer is 7 small calls per utterance: 1.03× → 1.01× real-time on the Pixel.
`force_gpu.txt` containing `dectx` restores the all-GPU placement for experiments.

Measured with LiteRT 2.1.6, decode included, app process warm:

* Pixel 8a (Tensor G3, Mali-G715), shipped placement: **~1.0× real-time** (8.8 s of
  speech in 8.75 s; 12.5 s in 13.1 s over 3 chunks). The gap to Adreno is per-step
  overhead (25 MB of packed-KV upload plus ~500 kernel dispatches per 78-MMAC step), not
  arithmetic. Pinning the flow-LM to CPU instead measures 1.24–1.60× here, but the LM
  stays on the GPU by policy.
* Samsung SM-S942Q (Snapdragon SM8850, Adreno): **4.3–5.0× real-time** (8.2 s in 1.63 s;
  13.0 s in 3.05 s over 3 chunks) — measured with the all-GPU placement and the split
  step+head graphs before the placement change; the dec-tx-on-CPU delta measured ~2% on
  the Pixel.

Whisper-transcribing the on-device WAVs reproduces the input text on both devices. The
KV-step `FULLY_CONNECTED` shapes are the class Mali rejects on LiteRT 2.1.3 and accepts
from 2.1.5, hence the 2.1.6 pin here.

## Re-authoring (numerically equivalent except one op)

| op | rewrite |
|---|---|
| erf-GELU | fitted odd tanh-polynomial `tanh(z(c1+z²(c3+z²(c5+z²c7))))`, max gelu err 7.1e-5 — ~15× closer than the classic tanh-GELU (which measurably shifted latents); z clamped to ±5.5 with RELUs so fp16 never sees big powers |
| interleaved RoPE | de-interleave permutation baked into the QKV projection rows (bit-exact: q and k share it, so q·k is unchanged); host feeds per-step cos/sin |
| rank-5 QKV reshape | slice the packed projection into thirds (GPU max rank is 4) |
| attention mask | runtime additive input, host-expanded to [1,16,1,513] (fp16-safe −1e4) |
| causal/banded mask (Mimi) | baked const additive bias [1,1,S,S], window 250 |
| ConvTranspose1d (×16 depthwise, SEANet) | ZeroStuffConvT1d (nearest-upsample × const stuff-mask + flipped grouped conv), streaming-trim reproduced by cropping |
| streaming conv state | constant left zero-pad (= fresh-state streaming, pad_mode constant) |
| ELU | `relu(x) − relu(1 − exp(min(x,0)))` (SELECT-free, exact) |
| LayerNorm (Mimi tx) | scaled-reduction form (exact; guards fp16 var against \|x\|≈59 activations) |
| LayerScale (Mimi tx) | baked into out_proj / linear2 rows |
| LSD time embeddings | s=0, t=1 are constants → folded into the flow head's cond bias; the +noise integration is in-graph |

## Three things that are easy to get wrong

**1. The Mimi decoder transformer's receptive field stacks.** Its sliding window is 250
positions, but with 2 layers a kept output needs 2·249 = 498 positions of exact left
context. Decoding in 512-position blocks with a 256-position overlap looks right and is
subtly wrong everywhere after the first block (corr 0.999, max err 0.5). Blocks here are
1024 positions (64 frames) sliding by 32 frames, keeping the right 512 — 512 ≥ 498 ⇒
exact (verified corr 1.000000 against full-sequence eager).

**2. The ×16 upsample reaches one frame back.** Kernel 32, stride 16: the first 16
positions of a block depend on the *previous* latent frame. Slot 0 of the dec_tx input is
that context frame; for the very first block the host passes the neutral latent
`−emb_mean/emb_std`, whose denormalized features are exactly zero — the algebraic
equivalent of "no frame". Feeding zeros instead injects `emb_mean` and corrupts the seam.

**3. The text prompt also "generates".** The reference implementation samples (and
discards) one latent right after prompting the text; the first real latent comes from the
BOS embedding, not from the last text position. Replaying text tokens per-step through the
same KV graph reproduces the batched prompt exactly (causal), but the BOS input is
`input_linear(bos_emb)`, a host-side constant.

## Build and run

For the best configuration, use the one-shot script:

```bash
cd pockettts/
scripts/reproduce_best.sh               # env + base graphs + int8 LM + AOT + shim
scripts/reproduce_best.sh push apk      # push to the device and install
```

By hand, that is:

```bash
cd pockettts/
# graphs + assets + parity (needs a pocket-tts clone on PYTHONPATH and the
# litert-torch conversion env; downloads the ungated english weights)
PYTHONPATH=/path/to/pocket-tts python scripts/build_pockettts.py all
# int8 flow-LM for the CPU (the single biggest win: 2.27x on the LM step)
PYTHONPATH=/path/to/pocket-tts PT_QUANT=dyn8_all python scripts/build_pockettts.py quant
# decoder transformer for the Tensor G5 NPU (AOT venv; optional)
python scripts/aot_tensor_g5.py pt_mimi_dec_tx_fp16
./scripts/fetch_google_tensor_dispatch.sh
./scripts/install_to_device.sh          # pushes scripts/out -> device files dir
./gradlew :app:installDebug
```

Environment this was built in, re-run 2026-09-18: Python 3.10, torch 2.12.1, litert-torch 0.9.3,
ai-edge-litert 2.1.6, ai-edge-quantizer 0.8.0, plus pocket-tts' own requirements (sentencepiece,
safetensors, huggingface_hub, scipy); pocket-tts clone at `001cf6e`. `all` takes about 3 minutes
on an M4 Max: it downloads the ungated `kyutai/pocket-tts-without-voice-cloning` weights, writes
every file the HF repo carries into `scripts/out/` (the re-run was byte-identical to the published
files — sha256 of all 20 LFS files matched), and prints the tflite-vs-eager numbers of the
Validation table below. Stages run individually (`flowlm`, `head`, `fused`, `dectx`, `deconly`,
`assets`, `pipeline`, `multistep`, `quant`); `PT_OUT=<dir>` redirects the output.

First launch before the push fails with "Missing pt_..." by design; run the app once,
push, relaunch. Everything (fp16 graphs + assets + 6 voices ≈ 225 MB) loads from the
app's external files dir.

## Validation

| check | result |
|---|---|
| flow-LM step vs eager (teacher-forced, 41 steps) | cond corr 1.000000, latent max\|d\| 1.4e-2 (all from the GELU polynomial; erf-GELU control: 1.7e-5) |
| flow head vs eager `lsd_decode` | max\|d\| 2.4e-7 |
| fused frame graph vs split step+head modules (12 free-run steps) | max\|d\| 0.0 (identical); tflite corr 1.000000 |
| dec_tx blocks vs full-sequence eager | corr 1.000000, max\|d\| 5.2e-4 |
| dec_tx + SEANet vs eager `decode_from_latent` | corr 1.000000, max\|d\| 2.0e-4 |
| full tflite pipeline vs eager, same noise | audio corr 0.997, identical EOS step |
| tokenizer port vs sentencepiece | 613/613 strings exact |
| on-device WAV → Whisper | input text reproduced |

## Voice cloning (not included)

Voice cloning needs the Mimi *encoder*, and its weights ship only in the **gated**
[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) repo — the ungated bundle
zeroes them. Accepting Kyutai's terms there unlocks the missing piece; the encoder is the
same SEANet re-authoring as the decoder (the mimi/ module in this zoo already has the
recipe), plus a per-step replay of the projected latents to warm the KV cache — the graphs
here would not change.
