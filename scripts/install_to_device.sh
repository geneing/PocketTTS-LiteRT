#!/bin/bash
# Push the Pocket TTS graphs + host assets into the app's external files dir.
# Build them first:  python scripts/build_pockettts.py all   (writes scripts/out/)
# Usage:             ./scripts/install_to_device.sh [dir-with-files]
#
# ADB=<path> overrides the adb binary. That matters on Windows, where adb lives
# in the SDK and the WSL adb server cannot see the device.
set -e
SRC="${1:-$(dirname "$0")/out}"
ADB="${ADB:-adb}"
DST="/sdcard/Android/data/com.pockettts/files"

FILES=(
  pt_flowlm_fused_fp16.tflite
  pt_mimi_dec_tx_fp16.tflite
  pt_mimi_deconly_fp16.tflite
  pt_embed_f16.bin
  pt_input_linear_f32.bin
  pt_bos_input_f32.bin
  pt_neutral_latent_f32.bin
  pt_tokenizer.tsv
  pt_voice_alba.bin
  pt_voice_marius.bin
  pt_voice_javert.bin
  pt_voice_charles.bin
  pt_voice_mary.bin
  pt_voice_eve.bin
)

# Optional graphs, pushed only when present:
#  * multi-step decode (`build_pockettts.py multistep`)
#  * int8 flow-LM variants (`build_pockettts.py quant`), picked by the
#    benchmark's lmGraph override
#  * AOT-compiled Tensor G5 variants (`aot_tensor_g5.py`), which the app picks
#    instead of the stock graph for an Accel.NPU placement.
EXTRA=(
  pt_flowlm_ms4_fp16.tflite
  pt_flowlm_ms8_fp16.tflite
  pt_flowlm_fused_dyn8_all.tflite
  pt_flowlm_fused_dyn8_body.tflite
  pt_flowlm_fused_dyn4_all.tflite
  pt_flowlm_fused_wo8_all.tflite
  pt_flowlm_fused_st8_body.tflite
  pt_flowlm_fused_fp16_g5.tflite
  pt_flowlm_ms4_fp16_g5.tflite
  pt_flowlm_ms8_fp16_g5.tflite
  pt_mimi_dec_tx_fp16_g5.tflite
)

# The app must have run once so Android creates its external files dir.
"$ADB" shell mkdir -p "$DST"
for f in "${FILES[@]}"; do
  echo "push $f"
  "$ADB" push "$SRC/$f" "$DST/$f" >/dev/null
done
for f in "${EXTRA[@]}"; do
  [ -f "$SRC/$f" ] || continue
  echo "push $f"
  "$ADB" push "$SRC/$f" "$DST/$f" >/dev/null
done
echo "done: $("$ADB" shell ls "$DST" | wc -l | tr -d ' ') files in $DST"
