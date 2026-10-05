# SEANet Tensor G5 experiment, phase-packed ConvTranspose

This experiment keeps the streaming SEANet graph on the Pixel 10 NPU and
replaces each zero-inserted transposed convolution with a phase-packed ordinary
convolution. For stride `s`, output sample `s*n+p` is computed directly from
the taps at `p, s+p, 2*s+p, ...`; the `s` phases are packed into output
channels. This avoids the large zero-filled activation in the earlier NPU
implementation.

The normal streaming graph and one-shot graph remain unchanged. The benchmark
uses a per-config streaming-graph filename and accelerator override, so this
experiment can be loaded without routing the unsupported one-shot SEANet graph
to the NPU.

## Export and compile

Use the pinned Pocket TTS clone and conversion environment described in
[`AGENTS.md`](../AGENTS.md):

```bash
PT_STREAM_W=512 PYTHONPATH="$(pwd)/references/pocket-tts:$(pwd)/scripts" \
    python scripts/export_seanet_phase.py
PT_OUT="$(pwd)/scripts/out" python scripts/aot_tensor_g5.py \
    pt_mimi_deconly_w512_phase_fp16
```

Change `PT_STREAM_W` and the graph name for the W=1024 experiment. The exporter
checks full-graph parity against the existing decoder and LiteRT CPU parity
before producing fp16 and AOT files.

## Pixel 10 results

The test paragraph has 201 words and 1,168 characters. It produces the same
60.72 seconds of audio and 759 frames on each path. Tests ran on the attached
Pixel 10 (Android 37, USB charging, screen on); they compare audio correlation,
not SNR. The benchmark saves 24 kHz mono WAVs under the app's
`power-benchmark/` external-files directory.

| Path | No-playback RTF | Playback stage time (LM / dec_tx / SEANet) | Audio correlation |
|---|---:|---:|---:|
| GPU reference, W=512 | 2.315× | 15.600 / 1.949 / 4.414 s | 1.000 |
| Phase-packed NPU, W=512 | 3.058× | 15.509 / 2.001 / 1.534 s | 1.000 |
| Phase-packed NPU, W=1024 | 2.678× | 15.631 / 2.080 / 3.491 s | 1.000 |

W=512 is the best result so far. Its SEANet runtime is about 65% lower than the
GPU reference, while full model-stage time drops from 21.963 s to 19.044 s. The
NPU compiler accepts all 155 operations in one partition: compilation took 7.8
seconds and produced a 9.4 MB G5 graph. W=1024 also compiles fully (10.8 seconds,
10.4 MB), but its larger window is slower on-device.

## Paired power reading, W=512

The test measured audio-only playback, then GPU-reference synthesis+playback,
then NPU synthesis+playback in one instrumentation session. Subtracting the
duration-scaled audio-only interval gave these system-domain totals:

| Path | CPU domains | GPU | TPU | Total |
|---|---:|---:|---:|---:|
| GPU reference | 27.292 J | 4.827 J | 1.544 J | 33.663 J |
| Phase-packed NPU | 25.983 J | ~0 J | 2.690 J | 28.673 J |

The measured total fell by 4.990 J (14.8%). The NPU path shifts work from GPU to
TPU; its extra TPU energy is smaller than the removed GPU energy. The raw GPU
delta was -0.028 J, which is monitor noise and is treated as zero here.
These are modeled system-domain counters while charging, not battery-capacity
measurements.

With the default placement, the CPU domain is largest, and the flow-LM is the
largest timed stage at about 15.6 s. The decoder transformer is about 2.0 s;
SEANet on GPU is about 4.4 s. With phase-packed SEANet on NPU, the LM remains the
largest stage and SEANet falls to about 1.5 s. CPU energy includes host work, so
the stage placement and timings are used together for attribution.

## Listening samples

The same seeded paragraph from the paired W=512 run is available locally:

- GPU reference: `scripts/out/seanet-npu-phase-reference.wav`
- W=512 NPU candidate: `scripts/out/seanet-npu-phase-candidate.wav`
- W=1024 NPU candidate: `scripts/out/seanet-npu-phase-w1024-candidate.wav`

The W=1024 case is slower than W=512 and was not included in the paired energy
comparison. Keep CPU-int8 SEANet as a separate experiment on
`codex/seanet-cpu-int8`; this NPU branch does not replace it.
