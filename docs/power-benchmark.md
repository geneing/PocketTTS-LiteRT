# Pixel TTS power benchmark

`PowerBenchmarkTest.longParagraphPlaybackPower` is the repeatable playback case
for comparing Pocket TTS placements and model variants. It uses a fixed 201-word
paragraph and seed, warms the delegates, then measures two intervals:

1. Play a cached reference waveform through `AudioTrack` (playback-only control).
2. Stream the same paragraph through Pocket TTS into `AudioTrack` while it is
   playing.

The test reads Android `PowerMonitor` counters before and after each interval.
It reports modeled CPU-cluster, GPU, and TPU energy consumers and the matching
ODPM rails when the device exposes them. The report also includes per-stage
model-run times, RTF from a separate no-audio streaming take, and waveform
correlation against that take. The audio-only control is scaled to the live
interval duration and subtracted from the full interval to estimate incremental
model energy.

## Run on the attached Pixel

```powershell
./gradlew.bat :app:installDebug :app:installDebugAndroidTest
adb shell am instrument -w `
  -e class com.pockettts.PowerBenchmarkTest#longParagraphPlaybackPower `
  com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -d -s PocketTTSPower:I
```

Install the models first with `scripts/install_to_device.sh` (or the PowerShell
equivalent `adb push` commands). The test needs the app's normal model files,
including `pt_flowlm_fused_dyn8_all.tflite`, `pt_mimi_dec_tx_fp16.tflite`, and
`pt_mimi_deconly_w512_fp16.tflite`.

The test plays the paragraph twice through the phone's media output. Run it at a
similar screen state and volume when comparing results. Keep the device on the
same power state and allow it to cool between long runs. Power-monitor energy
counters include both plugged-in and battery operation, so USB debugging can
remain connected; these readings are system-wide domain energy, not a direct
battery-capacity discharge measurement.

## Pixel 10 baseline

Measured on 2026-10-05 with Android API 37, 80% charge, USB power connected, and
the screen interactive. Default placement selected `lm:CPU dectx:NPU dec:GPU`.
The paragraph rendered 60.72 seconds of audio. Streaming synthesis took 24.715 s
(2.457× RTF); waveform correlation to the same-seed reference was 1.000.

| Stage | Placement | Model-run time | Share of measured stage time | Incremental domain energy estimate |
|---|---:|---:|---:|---:|
| Flow-LM | CPU | 15.171 s | 69.6% | 22.57 J across CPU clusters |
| Mimi decoder transformer (`dec_tx`) | TPU | 2.160 s | 9.9% | 1.55 J on TPU |
| SEANet waveform decoder | GPU | 4.481 s | 20.5% | 4.89 J on GPU |

The incremental estimates are the modeled consumer deltas for the full
synthesis-and-playback interval minus the duration-scaled audio-only control.
They sum to about 29.0 J across the three modeled compute domains. The CPU
Flow-LM domain accounts for about 78% of that estimate and is the dominant
consumer; SEANet is second, and `dec_tx` is smallest. Dividing each domain delta
by its stage time gives rough stage-average estimates of 1.49 W, 1.09 W, and
0.72 W respectively. These are attribution estimates, not isolated model-call
power measurements.

The hardware readings are domain-level, not model-call-level: CPU energy also
includes tokenization and host-side work, while GPU and TPU consumers include
other system work in those domains. Stage timing and accelerator placement
support the Flow-LM attribution, but the counters cannot isolate exact per-call
energy. Raw ODPM rail deltas are also logged; those rails can overlap modeled
consumers and must not be added to them. Incremental consumer deltas in this run
were CPU clusters 10.59 / 10.70 / 1.28 J, GPU 4.89 J, and TPU 1.55 J.

## CPU-int8 SEANet trial

`scripts/quantize_seanet.py` applies channelwise dynamic-range int8 weight
quantization to the full W=512 SEANet graph. It preserves float32 inputs and
outputs, produces 11 int8 weight tensors, and reduces the graph from 15.51 MiB
to 4.16 MiB. Run it in the documented Linux conversion environment:

```bash
python scripts/quantize_seanet.py \
  --input scripts/out/pt_mimi_deconly_w512.tflite \
  --output scripts/out/seanet_cpu_int8/pt_mimi_deconly_w512_dyn8.tflite
```

Push that file to
`/sdcard/Android/data/com.pockettts/files/pt_mimi_deconly_w512_dyn8.tflite`,
then run `PowerBenchmarkTest#longParagraphCpuInt8SeanetPower`. The test places
the candidate SEANet on CPU, keeps Flow-LM on CPU and `dec_tx` on NPU, and uses
the same-seed GPU output as its quality and audio-only reference.

The paired Pixel 10 run measures an audio-only interval, default GPU-SEANet
reference synthesis and playback, then CPU-int8 candidate synthesis and
playback. It uses the same 201-word paragraph, 60.72 s output, and seed. The
quality gate is waveform correlation (>=0.99); the saved WAVs are for listening
review. SNR is not used as an acceptance signal.

| Measure | Default GPU SEANet | CPU-int8 SEANet |
|---|---:|---:|
| No-playback inference | 24.605 s / 2.468× RTF | 24.359 s / 2.493× RTF |
| Playback stage time: Flow-LM / dec_tx / SEANet | 15.179 / 2.009 / 4.452 s | 15.409 / 1.985 / 8.273 s |
| Candidate-to-reference waveform correlation | — | 0.995 |
| Incremental CPU/0, CPU/1, CPU/2 | 8.945 / 2.855 / 2.264 J | 13.039 / 11.953 / 2.138 J |
| Incremental GPU / TPU | 4.263 / 1.374 J | -0.813 / 1.354 J |

The negative GPU delta is measurement noise and should be treated as roughly
zero. CPU-int8 keeps speed and passes the correlation gate, but moves energy
onto CPU: the modeled compute-domain estimate is about 19.7 J for the reference
and 28.5 J for the candidate (GPU noise floored at zero). The candidate SEANet
stage also takes about 1.86× as long as the GPU stage. Keep CPU-int8 available
as an experiment, but this Pixel 10 run does not support it as a battery-saving
placement. The WAVs are written under `scripts/out/` as
`seanet-cpu-int8-reference.wav` and `seanet-cpu-int8-candidate.wav` for the
user's listening comparison.
