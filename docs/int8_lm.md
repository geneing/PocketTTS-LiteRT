# int8 flow-LM on CPU (M6)

**Preferred configuration — `pt_flowlm_fused_dyn8_all.tflite` for the flow-LM on
CPU, decoder-transformer on the Tensor G5 NPU, SEANet on the GPU**
(`lm:CPU dectx:NPU dec:GPU`).

This is the fastest configuration measured on the Pixel 10 that passes the
by-ear test, and it is the only flow-LM the app loads (`PocketTts.LM`); the fp16
graph is a build byproduct now, not a fallback.

The end-to-end numbers below are the **one-shot** path. The app now streams, which
re-runs the same LM behind a sliding SEANet window and measures 3.1-3.3x RTF with
first audio at ~1.2 s — see `streaming.md`. The LM rows here are unaffected.

| | fp16 (`fused_fp16`) | int8 (`dyn8_all`) | |
|---|---|---|---|
| LM compute / frame | 19.59 ms | **8.65 ms** | **2.27x** |
| LM copy-in / frame | 1.33 ms | 1.50 ms | unchanged |
| LM readback / frame | 0.13 ms | 0.12 ms | unchanged |
| end-to-end RTF | 1.51x | **2.16x** | +43% |
| end-to-end, all-CPU | 0.83x | 0.99x | +19% |
| utterance | 3680 ms | **2564 ms** | 5.60 s audio |
| LM graph size | 161.4 MB | **81.7 MB** | 1.98x smaller |
| LM graph load | 274 ms | 150 ms | |

Measured on Pixel 10 (`frankel`), Tensor G5, PowerVR DXT-48-1536, Android 17;
95-char text, voice `alba`, seed `20260919`, 28 prompt + 70 generated frames.
Report: `bench/2026-09-19-pixel10-optim-tensor_g5_int8-4038d13.txt`.
Per-frame numbers are the `lm micro` line (128 frames, median of 3 runs).

## Why it works

The fp16 step is **DRAM-bandwidth-bound, not FLOP-bound**. The graph holds 84.44M
weight elements (337.8 MB in fp32, 168.9 MB in fp16) and the activations are
batch-1 and tiny, so each frame streams the whole weight set from DRAM:
337.8 MB / 19.6 ms ~= 17 GB/s, about what the phone's memory system delivers.
Quantizing the weights to int8 cuts that traffic 4x, and the measured compute
fell 2.27x — the gap is the parts int8 does not shrink (the 12 `BATCH_MATMUL`s,
the norms, the 25.2 MB packed-KV read, and the activations).

The prediction that this is bandwidth and not arithmetic is confirmed by the
`wo8_all` control: weight-only quantization produces the same 82 MB file but
inserts `DEQUANTIZE` and computes in float, and it saves **nothing** (1.04x).

## The recipe

Dynamic range, int8 weights, fp32 activations — equivalent to the
`dynamic_wi8_afp32()` preset (`dynamic_wi8c_afp32`):

```python
rm.add_dynamic_config(
    regex=".*",
    operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
    num_bits=8,
    granularity=qtyping.QuantGranularity.CHANNELWISE,
    algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT,
)
```

`ALL_SUPPORTED` resolves to the same 47 `FULLY_CONNECTED` ops as
`FULLY_CONNECTED` alone: the 12 `BATCH_MATMUL`s have no constant weight to
quantize, so nothing else changes.

**No calibration is needed, and the graph's I/O stays fp32.** The 7 inputs and
the single output are untouched, so the app's host protocol is identical to the
fp16 graph's — only the weight tensors change dtype. That is why this dropped in
without any change to `PocketTtsSynthesizer`'s step loop.

## Reproduce

```bash
# in WSL2, conversion venv active (see AGENTS.md)
PYTHONPATH=$(pwd)/references/pocket-tts PT_OUT=$(pwd)/scripts/out \
    PT_QUANT=dyn8_all python scripts/build_pockettts.py quant

./scripts/install_to_device.sh      # pushes it (it is in FILES)
./gradlew :app:installDebug         # it is the only LM the app loads
```

`PT_QUANT` takes a comma-separated list; omit it to build all eight variants.
The stage prints a per-slice parity line (eos / latent / new-k / new-v) against
the eager fp32 reference, and runs `opcheck` on the result.

## Variants tried, and why they lost

| variant | LM compute | RTF | size | ear test | verdict |
|---|---|---|---|---|---|
| **`dyn8_all`** | **8.65 ms** | **2.16x** | 82 MB | **good** | **preferred** |
| `dyn8_body` | 9.86 ms | 2.08x | 106 MB | good | correct but slower and larger: keeping the flow head fp32 costs 25 MB and 1.2 ms/frame for no audible gain |
| `wo8_all` | 18.90 ms | 1.56x | 82 MB | good | quality ceiling for int8 weights, but no speedup at all (float compute) |
| `dyn4_all` | 7.09 ms | 2.37x | 46 MB | **rejected** | fastest, but 4-bit weights collapse the trajectory (see below) |
| `st8_body` | 10.04 ms | 0.13x | 106 MB | broken | static FC-only; generation stops after **3 frames** |
| `dyn8_head` | — | — | 312 MB | — | diagnostic only: quantizing just the flow head leaves the body fp16 |
| `st8_all` | — | — | 86 MB | — | full-graph static; latent corr **-0.002** vs eager |
| `st16_all` | — | — | 86 MB | — | I/O becomes int16, which no host runner supports |

`dyn8_body` and `dyn8_head` localise the sensitivity: the flow head is the part
that cannot take int8 without cost (head-only quantization alone drops the
latent to corr 0.999906), but the body is where the bytes are, and the body alone
is enough.

### `dyn4_all` — why 4 bits is a cliff, not a slope

Same op scope and same 84.44M weight elements as `dyn8_all`; the difference is
4-bit weights at `BLOCKWISE_32` granularity (one fp16 scale per 32 weights)
instead of 8-bit at channelwise. Sizes confirm it: 42.2 MB of packed 4-bit
weights + ~5.3 MB of fp16 block scales = 47.8 MB.

- host parity vs eager: latent corr 0.990073 (vs 0.999813 for int8)
- device vs gold: corr 0.149, and the length drifts to 73 frames

~15 levels per block instead of 255, and the flow-LM feeds its latent back
autoregressively, so the per-step error compounds instead of averaging out.

### Static quantization is a dead end here

`st8_body` targets only `FULLY_CONNECTED` inside `FlowLMStep`, which does keep
the graph I/O fp32 — but it also quantizes **activations** to int8, and that is
strictly worse than dynamic range: latent corr 0.986172 against 0.999813, and on
device the eos logit is pushed past the -4.0 threshold so generation stops after
3 frames (0.24 s). `st8_all` (whole graph) additionally turns all 7 inputs and
the output int8 and collapses to corr -0.002.

Calibration data for the static runs comes from `quant_calibration()`, which
replays a free-run from the voice state so the calibrator sees the real offsets,
RoPE phases and accumulated-KV magnitudes. Two traps worth remembering if this is
ever revisited: `Quantizer.calibrate()` materialises the whole set (`len(list(...))`),
so samples must be subsampled rather than generated lazily (each carries ~50 MB
of KV); and the zero-noise cadence must be **coprime** with the subsample stride,
or every kept sample is a text-prompt step and the noise input calibrates to a
degenerate scale.

## Caveats

- **corr vs gold is not a quality metric for this LM.** The flow-LM is
  stochastic: a numerical perturbation yields a *different but valid*
  realization. The NPU LM scored 0.46 and was judged good; `dyn8_all` scores
  0.285 and was also judged good. Trajectory length is the honest automatic
  signal (`dyn8_all` produces exactly the reference 70 frames), and the ear is
  the decider.
- **The eos projection is the sensitive sub-op.** `out_eos` is a 1x1024
  `FULLY_CONNECTED` in the body whose quantization shifts the eos logit by up to
  0.66, and the threshold is -4.0. It is ~1K parameters, so quantizing it buys
  nothing — excluding it is the first thing to try if `dyn8_all` ever produces a
  truncated utterance.
- The 25.2 MB packed KV is still fp32 and is re-read every frame. Quantizing the
  cache (not just the weights) is the remaining bandwidth lever, but it needs a
  host-protocol change.

## Next levers, if more speed is wanted

- `dynamic_wi8c_hr_afp32` — Hadamard rotation before quantizing, which suppresses
  weight outliers and is the usual fix for a 4-bit collapse.
- `BLOCKWISE_64` int4 (fewer scales) or the `GPTQ` / `MSE` algorithms, all
  reachable through the `algorithm` key in `quant_recipe`.
- int8 weights **plus** an int8 KV cache.
- Combining with prompt prefill (`docs/multistep_lm_plan.md`, M3), which was
  worth 49.6 -> 2.4 ms/token on the prompt in the fp32 graph. The fused int8
  graph does carry the batched `prefill` signature, but on it the batched rows do
  not reproduce the per-token step (uncorrelated audio, short prompts truncated),
  so `usePrefill` is off and each prompt runs one fused step per token.

Audio exhibits for the ear test are in `bench/audio/` (`ptt_i8_*.wav`); the
joint CPU/GPU/NPU timing tables are in `RESULTS.md`.
