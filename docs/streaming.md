# Streaming generation (M7)

**The app streams.** The Mimi decoder runs behind the flow-LM and emits audio as
soon as it is decodable, instead of after the whole utterance. On the Pixel 10
that moves time-to-first-audio from ~2.6 s to **~1.2 s** while *improving*
end-to-end RTF from 2.15x to **3.1-3.3x**, because the SEANet decoder now runs
over a 512-position sliding window instead of the full 4096.

| | one-shot | stream `w=512` | stream `w=1024` | stream `w=2048` |
|---|---|---|---|---|
| utterance | 2576-2654 ms | **1691-1818 ms** | 1846-1964 ms | 1872-1939 ms |
| RTF | 2.15x | **3.08-3.31x** | 2.85-3.03x | 2.89-2.99x |
| time to first audio | ~2576 ms | **1179-1307 ms** | 1347-1463 ms | 1870-1937 ms |
| chunks | — | 3 | 2 | 1 |
| SEANet | 1377-1401 ms | **466-492 ms** | 635-660 ms | 663-686 ms |
| flow-LM | 855-870 ms | 858-897 ms | 854-873 ms | 847-877 ms |
| vs one-shot audio | — | corr 0.999999, max\|d\| 1.465e-03 | same | corr 1.000000, max\|d\| 0 |

Measured on Pixel 10 (`frankel`), Tensor G5, PowerVR DXT-48-1536, Android 17;
`lm:CPU dectx:NPU dec:GPU`, int8 flow-LM, 95-char text, voice `alba`, seed
`20260919`, 28 prompt + 70 generated frames = 5.60 s audio.
Report: `bench/2026-09-20-pixel10-optim-npu_int8_streaming-0b05d18.txt`.

The flow-LM is identical in every row — streaming changes only *when* the decoder
runs, so its 0.86 s is the floor for both metrics. **`w=512` is the app default**
(`PocketTtsSynthesizer.STREAM_W`).

## Why it is exact, not an approximation

The decoder is streaming-clean in both stages, which is why the concatenated
chunks match the one-shot audio rather than merely resembling it.

**The SEANet decoder is strictly causal.** Feeding the full 4096-position window
and a truncated one shows the first changed output sample is exactly `P*120` for
every truncation point `P` — zero lookahead — and the left receptive field is
~7.6 feature positions (constant in `P`). So a window fed at buffer offset 0
reproduces the full run on the region it covers, and keeping only
`STREAM_L = 16` positions of left context is a safety margin, not a fudge.
Confirmed bit-exact (`max|d| 0.0`) on the host for `L` = 128/256/512 and again
for every `L >= 8` with real dec_tx features at `W` = 512/1024/2048.

**dec_tx is already block-structured.** Its 64-frame blocks overlap by
`F_HOP = 32`, and only the region the overlap makes valid is kept, so running
blocks incrementally as frames arrive is the same arithmetic as running them all
at the end. Later blocks fire as soon as 32 more frames exist.

The first block is special: the decoder is *causal* (sliding-window transformer,
~31-frame left receptive field), so running a block with fewer than 64 real frames
and neutral padding yields the same output on the frames that exist. The streaming
path therefore runs the first block truncated and lets the (also strictly causal)
SEANet emit a **partial** window, so playback starts after `F_HOP = 32` frames
instead of `F_BLK = 64`.

There is **no ramp**. The emit step is the dec_tx hop, so every block the decoder
runs is one the one-shot decode also runs, from the same offset: frame 0 while the
prefix is at most `F_BLK`, then `F_HOP`-spaced sliding blocks. Streaming features
are therefore the one-shot features, and measured parity is at the graph's own
floor (`max|d| 1.5e-3`, `relDb` -59.0 dB on all three probe texts — the same
numbers `w=512` and `w=1024` give each other).

Anything finer moves block offsets off that chain. Stepping the decoder every 16
frames instead lands only every *other* block on an `F_HOP` multiple; those blocks
contribute the same frames from a different position and, through fp16
block-boundary rounding, the 246-character text fell from -58.9 dB to -50.6 dB
(`max|d|` 1.5e-3 -> 1.4e-2). The first-block size is the other such knob. The
earlier ramp emitted chunks of `8, 8, 16, 16, 16, 32, 32, 32, 64` frames, whose
cumulative targets (`8, 16, 32, 48, 64, 96, ...`) kept every block canonical and
started playback two targets earlier. It does not compose with the 31-frame window
hop, though: emitting is threshold-triggered, so each target needs its own dec_tx
block *and* its own SEANet run, and its small early targets cost more decoder work
than the points they save.

Each chunk has to outlast the wait for the next or playback starves. At `1.5x` a
frame of output is ~53 ms against ~36 ms of LM time, so a chunk of `h` frames
covers a hop of up to ~`1.5h`. With the step at `F_HOP = 32` every chunk covers
its own hop with a wide margin (and much more than a ramp starting at 8, whose
`8 -> 16` step was ~150 ms short on the Pixel 10).

The service also runs generation on a worker thread and drains a queue on the
framework thread (`PocketTtsService`): `audioAvailable()` blocks while the
playback buffer is full, and if that happened on the generation thread the LM
could not begin the next block until the current window had drained. With the
worker, chunk *n+1* is ready before chunk *n* finishes playing.

The result: the first audio chunk lands after the LM has produced 32 frames, not 64.

## Where the `max|d| 1.465e-03` comes from

It is **not** the windowing. `w=512` and `w=1024` produce **bit-identical**
audio to each other despite using different window sizes and a different number
of windows, so the sliding logic cannot be the source. What changes is the
delegate's convolution algorithm: above ~1024 elements the GPU picks a different
kernel, which reorders the fp32 accumulation. The 512/1024 graphs agree with each
other and differ from the 2048/4096 graphs by a global, sample-by-sample delta —
rms -73 dBFS, peak -52.5 dBFS against a signal peaking at 0.615, corr
0.9999995. `w=2048` is bit-exact on this GPU, which is why it is kept as the
conservative option.

That delta is inaudible and far tighter than deviations already accepted in this
project (`dyn8_all` itself scores corr 0.285 vs gold, the NPU LM 0.46 — see
`int8_lm.md`). The ear remains the decider; `w=2048` is the fallback if a
bit-exact take is ever wanted.

## Granularity

The first chunk of each text chunk is `F_HOP = 32` frames (2.56 s at 1x, 1.71 s at
1.5x); after that the SEANet emits whatever the decoder has produced, up to
`w - STREAM_L` positions (1.98 s at `w=512`), with the remainder last. Time to
first audio is dominated by the LM work for the prompt plus those 32 frames, not
by the decoder, so it is flat across sentence lengths (only the prompt grows with
the text). The 64-frame `dec_tx` graph is invoked once per `F_HOP` of new audio
after the prefix, plus one truncated block for the first chunk — the least
decoder work of any cadence.

## Reproduce

```bash
# in WSL2, conversion venv active (see AGENTS.md)
PYTHONPATH=$(pwd)/references/pocket-tts PT_OUT=$(pwd)/scripts/out \
    python scripts/build_pockettts.py stream     # PT_STREAM_W=512,1024,2048

./scripts/install_to_device.sh      # pushes all three windows
./gradlew :app:installDebug         # the app streams through w=512
```

`reproduce_best.sh stream` runs the same stage, and `PT_STREAM_W` overrides the
window list. The stage prints a prefix-parity line per window against the full
4096-position eager reference and runs `opcheck` on each export.

## How the app uses it

- `PocketTtsSynthesizer.synthesizeStream(text, voice, onChunk)` runs the same
  generation loop as `synthesize()` but hands each latent to a `StreamDecoder`,
  which drives dec_tx and the SEANet window as data becomes available.
  `synthesize()` is untouched and is still what the benchmark measures.
- If `pt_mimi_deconly_w512_fp16.tflite` is absent, `synthesizeStream` logs and
  falls back to `synthesize()` — one chunk, no streaming, no failure.
- `MainActivity`'s Generate button plays through an `AudioTrack` in
  `MODE_STREAM`, writing each chunk as it arrives; the blocking writes throttle
  the decoder to playback rate, so the status line reports time-to-first-audio
  rather than RTF.
- `Benchmarker` runs each window against the one-shot take and reports
  `first audio ... ms | N chunks` plus `maxAbsDiff`, which is how the
  windowing-vs-numerics question above was settled.

## Next levers

- A small (8- or 32-frame) dec_tx export to avoid the wasted padding compute in
  the first block; it would not change latency, only the decoder cost.
- Overlapping the SEANet window with the LM on separate threads (the LM is
  0.86 s of the 1.7 s and currently runs strictly ahead of the decoder).
- `w=2048` when bit-exactness matters more than the ~1.5x RTF difference.
