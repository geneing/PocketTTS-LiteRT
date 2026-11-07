#!/bin/bash
# Push the Pocket TTS graphs + host assets into the app's external files dir.
# Build them first:  python scripts/build_pockettts.py all   (writes scripts/out/)
# Usage:             ./scripts/install_to_device.sh [dir-with-files]
#
# ADB=<path> overrides the adb binary. That matters on Windows, where adb lives
# in the SDK and the WSL adb server cannot see the device. ADB_SERIAL=<serial>
# targets one device when several are attached (ANDROID_SERIAL does not cross
# into a Windows adb.exe launched from WSL).
set -e
SRC="${1:-$(dirname "$0")/out}"
ADB="${ADB:-adb}"
ADB_SERIAL="${ADB_SERIAL:-}"
DST="/sdcard/Android/data/com.pockettts/files"

adb() {
  if [ -n "$ADB_SERIAL" ]; then "$ADB" -s "$ADB_SERIAL" "$@"; else "$ADB" "$@"; fi
}

FILES=(
  pt_flowlm_fused_dyn8_all.tflite
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
#  * extra int8 flow-LM variants (`build_pockettts.py quant`) for the benchmark's
#    lmGraph override; the app's LM itself is pt_flowlm_fused_dyn8_all.tflite
#  * smaller-window SEANet decoders (`build_pockettts.py stream`) for streaming
#    generation; the app prefers pt_mimi_deconly_w512_fp16.tflite
#  * AOT-compiled Google Tensor NPU variants (`aot_tensor_npu.py`) and the
#    on-device-JIT FlowLM (`_jit`), which the app picks for an Accel.NPU
#    placement. Pixel 11 (Tensor G6) uses `_g6`/`_jit` suffixes.
EXTRA=(
  pt_flowlm_ms4_fp16.tflite
  pt_flowlm_ms8_fp16.tflite
  pt_flowlm_fused_dyn8_body.tflite
  pt_flowlm_fused_dyn4_all.tflite
  pt_flowlm_fused_wo8_all.tflite
  pt_flowlm_fused_st8_body.tflite
  pt_mimi_deconly_w512_fp16.tflite
  pt_mimi_deconly_w1024_fp16.tflite
  pt_mimi_deconly_w2048_fp16.tflite
  pt_flowlm_fused_fp16_contiguous_no_truncation_g6.tflite
  pt_flowlm_fused_fp16_contiguous_g6.tflite
  pt_flowlm_fused_fp16_g6.tflite
  pt_flowlm_ms4_fp16_g6.tflite
  pt_flowlm_ms8_fp16_g6.tflite
  pt_flowlm_fused_dyn8_all_g6.tflite
  pt_flowlm_fused_dyn8_all_contiguous_half_g6.tflite
  pt_flowlm_fused_dyn8_all_contiguous_jit.tflite
  pt_flowlm_step_fp16_g6.tflite
  pt_flow_head_fp16_g6.tflite
)

# The app must have run once so Android creates its external files dir.
adb shell mkdir -p "$DST"
for f in "${FILES[@]}"; do
  echo "push $f"
  adb push "$SRC/$f" "$DST/$f" >/dev/null
done
for f in "${EXTRA[@]}"; do
  [ -f "$SRC/$f" ] || continue
  echo "push $f"
  adb push "$SRC/$f" "$DST/$f" >/dev/null
done
# Extra voice caches (scripts/download_voices.py, scripts/create_voice.py) live in
# a voices/ subdirectory, one level below the model files, so a model refresh
# cannot clobber them and they need not be part of a model pack.
#
# The directory must be created BY THE APP, not here: on Android 11+ a
# mkdir/push under /sdcard/Android/data creates it owned by `shell` with
# `drwxrws---`, which the app process cannot traverse, so the voices in it are
# invisible to it. The app makes it on startup (VoiceCatalog.ensureDir), so run it
# once before the first voice push. Anything left in the root by an older install
# is pushed too, so those voices keep working.
VOICES_DIR="voices"
if ! adb shell "[ -d '$DST/$VOICES_DIR' ]"; then
  echo "note: $DST/$VOICES_DIR does not exist yet; launch the app once so it" >&2
  echo "      creates it app-owned, then re-run this script." >&2
fi
for f in "$SRC"/pt_voice_*.bin; do
  [ -f "$f" ] || continue
  case " ${FILES[*]} " in *" $(basename "$f") "*) continue ;; esac
  echo "push $VOICES_DIR/$(basename "$f")"
  adb push "$f" "$DST/$VOICES_DIR/$(basename "$f")" >/dev/null
done
if [ -d "$SRC/$VOICES_DIR" ]; then
  for f in "$SRC/$VOICES_DIR"/pt_voice_*.bin; do
    [ -f "$f" ] || continue
    echo "push $VOICES_DIR/$(basename "$f")"
    adb push "$f" "$DST/$VOICES_DIR/$(basename "$f")" >/dev/null
  done
fi
echo "done: $(adb shell ls "$DST" | wc -l | tr -d ' ') files in $DST, " \
     "$(adb shell ls "$DST/$VOICES_DIR" 2>/dev/null | wc -l | tr -d ' ') in $VOICES_DIR"
