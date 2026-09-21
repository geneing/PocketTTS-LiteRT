#!/usr/bin/env bash
#
# Reproduce the best measured Pocket TTS configuration in one run.
#
# The configuration (Pixel 10 / Tensor G5, 3.2x real-time -- docs/streaming.md):
#
#   flow-LM   pt_flowlm_fused_dyn8_all.tflite   CPU  dynamic-range int8 weights
#   dec_tx    pt_mimi_dec_tx_fp16_g5.tflite     Tensor G5 NPU (AOT-compiled)
#   SEANet    pt_mimi_deconly_w512_fp16.tflite  GPU, sliding 512-position window
#
# Usage:
#   scripts/reproduce_best.sh [stage ...]
#
# Stages, run in this order when none are named (default: env build quant stream aot shim):
#   env     check the toolchain and report what is missing
#   build   export the base graphs + host assets      build_pockettts.py all
#   quant   export the int8 flow-LM                   PT_QUANT=dyn8_all
#   stream  export the sliding-window SEANet decoders build_pockettts.py stream
#   aot     AOT-compile dec_tx for the Tensor G5 NPU  aot_tensor_g5.py
#   shim    fetch libLiteRtDispatch_GoogleTensor.so   fetch_google_tensor_dispatch.sh
#   push    push every file the app loads to the device
#   apk     ./gradlew :app:installDebug
#
# So `scripts/reproduce_best.sh` writes the weights, and
# `scripts/reproduce_best.sh push apk` gets them onto a phone.
#
# Environment (all optional):
#   PT_VENV      conversion venv        default ~/pockettts-conv/.venv
#   PT_AOT_VENV  AOT venv               default ~/pockettts-aot/.venv
#   PT_REFS      pocket-tts clone       default <repo>/references/pocket-tts
#   PT_OUT       where graphs are written   default scripts/out
#   PT_QUANT     int8 variant to build  default dyn8_all
#   ADB          adb binary             default adb
#
# Linux or WSL2 only: litert-converter has no Windows wheel, so the export
# cannot run on a native Windows host. Run this from WSL2 (see AGENTS.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
REPO="$ROOT"
OUT="${PT_OUT:-$ROOT/scripts/out}"
REFS="${PT_REFS:-$REPO/references/pocket-tts}"
VENV="${PT_VENV:-$HOME/pockettts-conv/.venv}"
AOT_VENV="${PT_AOT_VENV:-$HOME/pockettts-aot/.venv}"
ADB="${ADB:-adb}"
QUANT="${PT_QUANT:-dyn8_all}"

# PT_OUT must stay a POSIX path; a Windows-style one becomes a literal
# directory named "C:..." when the tools run under WSL.
export PT_OUT="$OUT"

say() { printf '\n=== %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }

# The files the best configuration actually loads, in push order. Kept in step
# with PocketTtsSynthesizer's constants and install_to_device.sh.
BEST_FILES=(
  pt_flowlm_fused_dyn8_all.tflite
  pt_mimi_dec_tx_fp16_g5.tflite
  pt_mimi_deconly_w512_fp16.tflite
  pt_mimi_deconly_fp16.tflite
  pt_embed_f16.bin
  pt_input_linear_f32.bin
  pt_bos_input_f32.bin
  pt_neutral_latent_f32.bin
  pt_tokenizer.tsv
  pt_voice_alba.bin
)

stage_env() {
  say "env"
  local bad=0
  if [ -d "$REFS" ]; then
    echo "ok   pocket-tts clone   $REFS ($(git -C "$REFS" rev-parse --short HEAD 2>/dev/null || echo '?'))"
  else
    echo "MISS  pocket-tts clone  $REFS"
    echo "      git clone https://github.com/kyutai-labs/pocket-tts.git \"$REFS\""
    echo "      git -C \"$REFS\" checkout 001cf6e"
    bad=1
  fi
  if [ -x "$VENV/bin/python" ]; then
    echo "ok   conversion venv    $VENV"
  else
    echo "MISS  conversion venv   $VENV  (see AGENTS.md / requirements-convert.txt)"
    bad=1
  fi
  if [ -x "$AOT_VENV/bin/python" ]; then
    echo "ok   AOT venv           $AOT_VENV"
  else
    echo "MISS  AOT venv          $AOT_VENV"
    echo "      only the dec_tx-on-NPU stage needs it; without it that graph stays"
    echo "      on CPU and the rest of the configuration is unaffected"
  fi
  command -v curl >/dev/null && echo "ok   curl" || { echo "MISS  curl"; bad=1; }
  command -v unzip >/dev/null && echo "ok   unzip" || { echo "MISS  unzip"; bad=1; }
  echo "     output dir          $OUT"
  if [ "$bad" = 1 ]; then
    echo
    echo "the missing pieces above are fatal; set PT_REFS / PT_VENV to point elsewhere"
    return 1
  fi
}

stage_build() {
  say "build: base graphs + host assets"
  PYTHONPATH="$REFS" "$VENV/bin/python" "$ROOT/scripts/build_pockettts.py" all
}

stage_quant() {
  say "quant: int8 flow-LM ($QUANT)"
  PYTHONPATH="$REFS" PT_QUANT="$QUANT" \
    "$VENV/bin/python" "$ROOT/scripts/build_pockettts.py" quant
}

stage_stream() {
  say "stream: smaller-window SEANet decoders"
  # The app streams: it runs the SEANet decoder over a sliding window as the LM
  # produces frames, so audio starts ~1.2s in instead of after the whole
  # utterance. The 512 window is the app default; 1024/2048 are kept so the
  # benchmark can compare them.
  PYTHONPATH="$REFS" \
    "$VENV/bin/python" "$ROOT/scripts/build_pockettts.py" stream
}

stage_aot() {
  say "aot: dec_tx for the Tensor G5 NPU"
  if [ ! -x "$AOT_VENV/bin/python" ]; then
    warn "no AOT venv at $AOT_VENV -- skipping, dec_tx will run on CPU"
    return 0
  fi
  # Only the decoder transformer is worth compiling: the SEANet body fails in
  # the backend compiler and the flow-LM is faster on CPU than on the NPU.
  "$AOT_VENV/bin/python" "$ROOT/scripts/aot_tensor_g5.py" pt_mimi_dec_tx_fp16
}

stage_shim() {
  say "shim: Google Tensor dispatch library"
  bash "$ROOT/scripts/fetch_google_tensor_dispatch.sh"
}

stage_push() {
  say "push: device files dir"
  ADB="$ADB" bash "$ROOT/scripts/install_to_device.sh" "$OUT"
}

stage_apk() {
  say "apk: build + install"
  ( cd "$ROOT" && ./gradlew :app:installDebug )
}

stage_summary() {
  say "artifacts for the best configuration"
  local missing=0
  for f in "${BEST_FILES[@]}"; do
    if [ -f "$OUT/$f" ]; then
      printf '  %8.1f MB  %s\n' \
        "$(awk "BEGIN{print $(stat -c%s "$OUT/$f")/1000000}")" "$f"
    else
      printf '  %8s     %s  (MISSING)\n' "-" "$f"
      missing=1
    fi
  done
  echo
  if [ "$missing" = 0 ]; then
    echo "all present. run the app and it picks this configuration automatically:"
    echo "  the flow-LM prefers the int8 graph, dec_tx the _g5 graph when the"
    echo "  dispatch shim is installed, and SEANet streams through the 512-position"
    echo "  window graph on the GPU."
    echo "  scripts/reproduce_best.sh push apk     # to get it onto a phone"
  else
    warn "some artifacts are missing -- see the stage output above"
    return 1
  fi
}

ALL=(env build quant stream aot shim)
if [ "$#" -eq 0 ]; then
  stages=("${ALL[@]}")
else
  stages=("$@")
fi

for s in "${stages[@]}"; do
  case "$s" in
    env)     stage_env ;;
    build)   stage_build ;;
    quant)   stage_quant ;;
    stream)  stage_stream ;;
    aot)     stage_aot ;;
    shim)    stage_shim ;;
    push)    stage_push ;;
    apk)     stage_apk ;;
    summary) stage_summary ;;
    *)       echo "unknown stage: $s" >&2; exit 2 ;;
  esac
done

# Report the artifact list after a full produce-the-weights run.
if [ "$#" -eq 0 ]; then
  stage_summary || true
fi
