# FlowLM selective W8/A16: FFN12 Pixel long pairs

The full static W8/A16 and all-FC W8/A16 recipes failed first-step tensor
parity (see the companion `2026-10-06-flow-g5-static-w8a16-candidate.md`).
This follow-up quantizes only large FlowLM feed-forward network (FFN) dense
matrices. It retains float32 attention projections, K/V handling, EOS and
flow head, and all Android-facing tensors. Both selective candidates passed
the first-step host screen and completed a 32-step Alba free-run comparison.
The FFN12 variant was AOT compiled for Tensor G5 and measured in two reversed
long-utterance pairs on Pixel 10. It ran completely on the NPU partition but
remained slower than the CPU dynamic INT8 control. Spoken-text completion and
listening quality have not been reviewed.

## Selection and reproduction

The position-major source has 47 `FULLY_CONNECTED` operations. Each of its six
FlowLM layers has one `[4096,1024]` FFN expansion matrix and one `[1024,4096]`
FFN output matrix. `static16_ffn12` quantizes both matrices in all six layers,
covering 12/47 dense operations and 50,331,648/84,444,160 dense weights
(59.6%). `static16_ffn6` quantizes only the six FFN output matrices, covering
6/47 operations and 25,165,824/84,444,160 dense weights (29.8%).

The CLI discovers the exact source output tensor names by weight shape and
restricts the AEQ recipe to those names. It requires exactly 12 or six unique
matches and verifies the exported INT8 weight count and shapes. AEQ's
`get_op_scope` adds a trailing semicolon to the output name; the exact-match
regex includes it. The first attempted selector omitted it and produced an
unquantized source copy; the coverage assertion caught that result before
parity testing. Only the corrected candidates below count as experiments.

Run from branch `codex/flowlm-g5-static-w8a16` in the pinned WSL environment
with source `pt_flowlm_fused_fp32_contiguous.tflite` in `PT_OUT`. Observed
versions: PyTorch 2.12.1+cpu, LiteRT Torch 0.9.3, AI Edge LiteRT 2.1.6,
AI Edge Quantizer 0.8.0. The local ignored `scripts/out/` contains source
and CPU dyn8 symlinks; generated artifacts are not committed.

```bash
PYTHONPATH=/path/to/references/pocket-tts PT_OUT=/path/to/scripts/out \
  python scripts/quantize_flowlm_g5.py --recipe static16_ffn12 \
  --calibration-voices alba --calibration-seed 11 \
  --calibration-samples 8 --calibration-run 96 --steps 32 \
  --compare-dyn8 /path/to/pt_flowlm_fused_dyn8_all.tflite

PYTHONPATH=/path/to/references/pocket-tts PT_OUT=/path/to/scripts/out \
  python scripts/quantize_flowlm_g5.py --recipe static16_ffn6 \
  --calibration-voices alba --calibration-seed 11 \
  --calibration-samples 8 --calibration-run 96 --steps 32 \
  --compare-dyn8 /path/to/pt_flowlm_fused_dyn8_all.tflite
```

Both use profiler calibration from Alba's pinned voice state, free-running
offsets 126..222 and retaining eight samples at 126, 138, 150, 162, 174,
186, 198 and 210. Each sample includes live embedding, RoPE, mask, K/V,
and noise inputs. The rollout starts from the same Alba voice at offset 126
and uses noise seed 3. Each graph feeds back its own latent and emitted K/V
rows after the first step; eager FP32 and CPU dyn8 each have independent
feedback loops.

## Graph interface and actual quantization

Both candidates have seven float32 inputs: embedding `[1,1,1024]`, cosine and
sine `[1,1,1,64]` each, mask `[1,16,1,513]`, position-major K/V each
`[1,512,96,64]`, and noise `[1,32]`. Their one float32 output is `[1,12321]`
for EOS, 32 latent values, 6144 new K values and 6144 new V values.
The current Android position-major host protocol therefore remains valid.

| Candidate | INT8 FC weights | INT16 tensors | Quantize / dequantize ops | Float32 FC weights |
|---|---:|---:|---:|---:|
| FFN12 | 12/47 (six each `[4096,1024]` and `[1024,4096]`) | 24 | 12 / 12 | 35/47 |
| FFN6 | 6/47 (all `[1024,4096]`) | 12 | 6 / 6 | 41/47 |

All QKV projection, attention output, EOS, and flow-head FC weights remain
float32 in both variants. The quantized matrix activation and output tensors
are INT16; external cache and result tensors remain float32.

## Host tensor quality

The first four autoregressive steps compare each graph with the eager FP32
trajectory. CPU dyn8 first-four values use the same Alba/noise setup and were
recorded in `2026-10-06-flow-g5-static-int8-candidate.md`; its complete
32-step summary was re-measured by this script using the graph's step
signature (index 1).

| Step | FFN12 latent corr | FFN6 latent corr | CPU dyn8 latent corr | FFN12 EOS abs delta | FFN6 EOS abs delta | CPU dyn8 EOS abs delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.99980462 | 0.99981426 | 0.99981317 | 0.3122 | 0.3932 | 0.6556 |
| 1 | 0.98985330 | 0.99002487 | 0.99980694 | 0.6995 | 0.5963 | 0.2444 |
| 2 | 0.95310265 | 0.95876089 | 0.99961405 | 0.8664 | 0.5801 | 0.2223 |
| 3 | 0.95751255 | 0.95882887 | 0.99950914 | 1.033 | 1.107 | 0.1625 |

| Step | FFN12 latent max delta | FFN6 latent max delta | FFN12 K/V max delta | FFN6 K/V max delta | CPU dyn8 K/V max delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.08976 | 0.08998 | 0.1021 | 0.09839 | 0.09495 |
| 1 | 0.8354 | 0.7801 | 0.5876 | 0.1487 | 0.1647 |
| 2 | 0.9927 | 0.9678 | 0.5769 | 0.5939 | 0.2729 |
| 3 | 0.7158 | 0.7544 | 0.7887 | 0.5051 | 0.3640 |

| 32-step self-fed result | FFN12 | FFN6 | CPU dyn8 |
|---|---:|---:|---:|
| Minimum latent correlation | 0.66987199 | 0.39914435 | -0.12466197 |
| Maximum latent absolute delta | 2.486 | 4.388 | 6.044 |
| Maximum EOS absolute delta | 1.931 | 4.412 | 2.781 |
| Maximum emitted K/V absolute delta | 2.663 | 2.772 | 4.727 |

The selective results show much better first-step behavior than the full or
all-FC static W8/A16 graphs. The CPU dyn8 model has closer step-1..3 latent
agreement, while its 32-step free-run diverges more in this one seed; free-run
correlation is a trajectory diagnostic, not a speech quality verdict. FFN12
retains more INT8 weight coverage and had the stronger 32-step result here.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `pt_flowlm_fused_fp32_contiguous.tflite` source | 338,170,404 | `908a5c9f9487d5ba44b6fe4e0d79f8921a626781fdf222a5c5e08ee5267326fa` |
| `pt_flowlm_fused_st16_ffn12_contiguous.tflite` | 187,545,872 | `3f045abcf256ff1716cc497c26abd2f49a4c880d2ca9c6c990a3306a3e5336f0` |
| `pt_flowlm_fused_st16_ffn6_contiguous.tflite` | 262,745,632 | `ad2ebba4147cea2253ba7c371de3c013c360aea54ae89fe3cc3cc715e3d10211` |
| `pt_flowlm_fused_st16_ffn12_contiguous_no_truncation_g5.tflite` | 178,697,824 | `183a71a2475e6fb68ab626788f8f2189a85d9fb661bcca8c8637e19ce2906d12` |

## Tensor G5 AOT result for FFN12

The separate pinned AOT environment has `ai-edge-litert==2.2.0` and
`ai-edge-litert-sdk-google-tensor==2.2.0`. Only FFN12 was compiled, using:

```bash
PT_OUT=/path/to/scripts/out .venv-aot/bin/python scripts/aot_tensor_g5.py \
  pt_flowlm_fused_st16_ffn12_contiguous --truncation no_truncation
```

The compiler completed in **34.0 s** and reported `Subgraph 0 fully
compiled: 694 / 694 ops offloaded to 1 partitions`. The exported file above
contains one `serving_default` signature and one public `DISPATCH_OP`.
Interpreter inspection found the same seven float32 inputs and one float32
`[1,12321]` output, including position-major K/V inputs `[1,512,96,64]`.
The compiled program is opaque, so the host cannot establish which internal
weight precision the vendor program ultimately uses or execute its numerical
path without Tensor G5.

The compiled artifact is at
`build/flowlm-worktrees/option12-static-w8a16/scripts/out/pt_flowlm_fused_st16_ffn12_contiguous_no_truncation_g5.tflite`.
Its logical Android `npuGraph` name is
`pt_flowlm_fused_st16_ffn12_contiguous_no_truncation.tflite` with
`npuPositionMajorCache=true`. The paired Pixel results appear below.

## Pixel long-pair procedure

The AOT artifact was pushed into the app's external model directory after
the previous device experiment finished. `PocketTts.g5Variant`
appends `_g5` to the logical `npuGraph` name. The app and test APKs must be
installed, but a model-only change needs no APK rebuild. Use direct
instrumentation; `:app:connectedDebugAndroidTest` uninstalls the app and
removes pushed model files.

```powershell
$adbExe = 'C:\Users\genei\AppData\Local\Android\Sdk\platform-tools\adb.exe'
$pixelSerial = '57220DLCR002R6'
$candidateFile = 'I:\Android_Projects\PocketTTS-LiteRT\build\flowlm-worktrees\option12-static-w8a16\scripts\out\pt_flowlm_fused_st16_ffn12_contiguous_no_truncation_g5.tflite'
& $adbExe -s $pixelSerial push $candidateFile /sdcard/Android/data/com.pockettts/files/pt_flowlm_fused_st16_ffn12_contiguous_no_truncation_g5.tflite
& $adbExe -s $pixelSerial shell sha256sum /sdcard/Android/data/com.pockettts/files/pt_flowlm_fused_st16_ffn12_contiguous_no_truncation_g5.tflite
```

The checksum returned by the device was
`183a71a2475e6fb68ab626788f8f2189a85d9fb661bcca8c8637e19ce2906d12`.
The harness defaults to Alba/seed 42 for the long text, but both are passed
explicitly here. It uses the shipped CPU dynamic INT8 graph as the CPU arm,
with the same NPU Mimi transformer and GPU SEANet placements on both arms.
Run one instrumentation method at a time, first NPU then CPU and then reversed:

```powershell
& $adbExe -s $pixelSerial shell am instrument -w `
  -e class com.pockettts.FlowLmHarnessTest#npuResidentCacheSpeechPair `
  -e npuSliceCache true -e npuResidentCache false `
  -e npuPositionMajorCache true -e verifyNpuSliceRows false `
  -e npuGraph pt_flowlm_fused_st16_ffn12_contiguous_no_truncation.tflite `
  -e referenceGraph pt_flowlm_fused_dyn8_all.tflite `
  -e workload long -e voice alba -e seed 42 -e energyRepeats 1 `
  -e order npu-cpu -e aotReport G5-694of694ops-1partition-LiteRT2.2.0 `
  com.pockettts.test/androidx.test.runner.AndroidJUnitRunner

& $adbExe -s $pixelSerial shell am instrument -w `
  -e class com.pockettts.FlowLmHarnessTest#npuResidentCacheSpeechPair `
  -e npuSliceCache true -e npuResidentCache false `
  -e npuPositionMajorCache true -e verifyNpuSliceRows false `
  -e npuGraph pt_flowlm_fused_st16_ffn12_contiguous_no_truncation.tflite `
  -e referenceGraph pt_flowlm_fused_dyn8_all.tflite `
  -e workload long -e voice alba -e seed 42 -e energyRepeats 1 `
  -e order cpu-npu -e aotReport G5-694of694ops-1partition-LiteRT2.2.0 `
  com.pockettts.test/androidx.test.runner.AndroidJUnitRunner
```

Each run writes its report and NPU/CPU WAVs under the device app external
files `flowlm-npu-slice-position-major/speech-<timestamp>/` directory. The
`aotReport` argument records compiler coverage alongside the on-device graph
checksum, backend, power monitors, elapsed time, first audio and stage timing.

## Pixel long-pair results

Both direct instrumentation commands above passed as single methods. Device
serial `57220DLCR002R6` identified Pixel 10 (`frankel`), Android 17,
fingerprint `google/frankel/frankel:17/CP3A.260905.009/16091614:user/release-keys`.
The phone's SHA-256 of the installed AOT model matched the artifact table.
Both reports show `g5HighPerformance=false`, NPU arm placement
`lm:NPU dectx:NPU dec:GPU`, and CPU control `lm:CPU dectx:NPU dec:GPU`.
The model remained the sole 694/694-op Tensor G5 AOT partition. The fixed
long text, Alba voice, seed 42, and `energyRepeats=1` were the same in both
orders. The harness measured one synthesis interval per arm after its warmup
and same-workload audio-only sample.

| Order | NPU frames / speech | CPU frames / speech | Engine inference NPU / CPU | NPU/CPU | First audio NPU / CPU | Thermal status |
|---|---:|---:|---:|---:|---:|---:|
| NPU then CPU | 769 / 61.52 s | 759 / 60.72 s | 32.857 / 24.834 s | 1.323x | 1.510 / 1.122 s | 0 |
| CPU then NPU | 769 / 61.52 s | 759 / 60.72 s | 32.292 / 24.141 s | 1.338x | 1.687 / 1.133 s | 0 |
| Two-order mean | 769 / 61.52 s | 759 / 60.72 s | 32.575 / 24.488 s | **1.330x** | 1.599 / 1.128 s | 0 |

Mean engine inference real-time factors were 0.529 for FFN12 NPU and 0.403 for
CPU dynamic INT8. The NPU produced ten more frames (0.80 s) in both orders;
frame count alone does not establish that every word was spoken. The paired
speed target was missed in both orders.

| Order and arm | LM input ms | LM run ms | LM read ms | Native cache row copy ms | Mimi dec-tx ms | Mimi SEANet ms | LM load ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| NPU then CPU: FFN12 NPU | 1,281 | 17,100 | 2,292 | 1,767 | 3,274 | 4,661 | 162 |
| NPU then CPU: CPU dyn8 | 2,382 | 14,970 | 131 | n/a | 1,679 | 4,523 | 1,775 |
| CPU then NPU: FFN12 NPU | 1,156 | 16,914 | 2,198 | 1,688 | 3,442 | 4,693 | 233 |
| CPU then NPU: CPU dyn8 | 2,259 | 14,850 | 117 | n/a | 1,495 | 4,441 | 1,807 |

The NPU LM run stage averaged 17.007 s versus 14.910 s on CPU. NPU output
readback averaged 2.245 s versus 0.124 s; NPU cache row copies averaged
1.728 s and are a component of its native cache path. The NPU saved about
1.102 s in LM input staging yet had longer LM execution, readback, and Mimi
dec-tx times. The candidate had 1,117 LM steps versus 1,107 for CPU because
of the ten extra frames. NPU model load took 162/233 ms versus CPU dynamic
INT8 1,775/1,807 ms; load is separate from the synthesis interval.

The harness estimates incremental energy by measuring Android power monitors
during synthesis, measuring audio-only playback of the same workload, then
subtracting audio-only energy scaled by elapsed duration. The selected
aggregate-domain values below are joules for each measured synthesis interval,
not a model-only power measurement. The full rail output is in each pulled
`report.txt`.

| Order and arm | CPU/0 J | CPU/1 J | CPU/2 J | GPU/0 J | TPU/1 J | Display J |
|---|---:|---:|---:|---:|---:|---:|
| NPU then CPU: FFN12 NPU | 0.742 | 2.392 | 0.027 | 5.001 | 12.108 | -0.116 |
| NPU then CPU: CPU dyn8 | 10.577 | 11.453 | 0.067 | 4.900 | 1.418 | 0.330 |
| CPU then NPU: FFN12 NPU | 0.405 | -1.252 | 0.189 | 5.208 | 12.257 | -0.166 |
| CPU then NPU: CPU dyn8 | 13.284 | 15.088 | 6.447 | 5.119 | 1.862 | 0.500 |

The negative incremental values and the CPU/2 control change from 0.067 to
6.447 J show the limit of a single-repeat, duration-scaled subtraction.
TPU/1 was about 12.1 J for NPU versus 1.4-1.9 J for CPU, while CPU domain
energy moved the other way. These pairs do not support a stable total-energy
claim without more power repeats, and latency already misses the speed gate.

| Order | Waveform correlation | SNR dB | High-band error dB | Reference HNR dB | FFN12 HNR dB |
|---|---:|---:|---:|---:|---:|
| NPU then CPU | 0.072400 | 0.022825 | 0.032678 | 0.795331 | 0.368675 |
| CPU then NPU | 0.072428 | 0.022842 | 0.032723 | 0.795849 | 0.368660 |

Free-running waveform correlation is diagnostic and cannot by itself reject
speech. Both runs wrote 61.52 s FFN12 and 60.72 s CPU WAVs. Listening and
spoken-text completeness remain unverified.

Reports and all four WAVs were pulled into ignored local directories in this
worktree before releasing the phone:

| Order | Pulled directory |
|---|---|
| NPU then CPU | `build/flowlm-w8a16-device/npu-first/speech-20261006-164542-399/` |
| CPU then NPU | `build/flowlm-w8a16-device/cpu-first/speech-20261006-165020-478/` |

Each directory contains `report.txt`, `npu.wav` and `cpu.wav`.
