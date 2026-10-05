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
model-run times, RTF from a separate synthesis-only take, and waveform
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

## Initial Pixel 10 baseline

Measured on 2026-10-05 with Android API 37, 80% charge, USB power connected, and
the screen interactive. Default placement selected `lm:CPU dectx:NPU dec:GPU`.
The paragraph rendered 60.72 seconds of audio; synthesis-only RTF was 1.952×,
waveform correlation was 1.000, and SNR was 59.7 dB.

| Stage | Placement | Model-run time | Share of measured stage time | Incremental domain energy estimate |
|---|---:|---:|---:|---:|
| Flow-LM | CPU | 15.643 s | 71.0% | 11.45 J across CPU clusters |
| Mimi decoder transformer (`dec_tx`) | TPU | 1.989 s | 9.0% | 1.41 J on TPU |
| SEANet waveform decoder | GPU | 4.407 s | 20.0% | 4.75 J on GPU |

The incremental estimates are the modeled consumer deltas for the full
synthesis-and-playback interval minus the duration-scaled audio-only control.
They sum to about 17.6 J across the three modeled compute domains; the CPU
Flow-LM is the largest contributor in this run. The hardware readings are
domain-level, not model-call-level: CPU energy also includes tokenization and
host-side work, while the GPU and TPU consumers include any other system work in
those domains. Use stage timing and the placement mapping alongside energy
before assigning a change to a graph.

The modeled consumer deltas were CPU clusters 8.17 / 3.05 / 0.23 J, GPU 4.75 J,
and TPU 1.41 J. Raw ODPM rail deltas are also logged, but those rails can overlap
the modeled consumers and must not be added to them.
