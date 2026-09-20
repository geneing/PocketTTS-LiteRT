#!/bin/bash
# Fetch the Google Tensor NPU dispatch shim into app/src/main/jniLibs/arm64-v8a/.
#
# A Tensor-G5 AOT model is executed by a dispatch shim that dlopen()s the
# device's own EdgeTPU runtime (/vendor/lib64/libedgetpu_litert.so). The shim is
# NOT in the LiteRT Maven AAR -- the AAR ships only libLiteRt.so and the OpenCL
# accelerator -- it comes from the `litert_npu_runtime_libraries.zip` asset on
# the matching LiteRT GitHub release. Version matching is the whole game: the
# shim, the Android `litert` runtime and the AOT compiler must all be the same
# LiteRT release, or the model fails to initialize (or silently misses the NPU).
#
# This app pins litert:2.2.0 and compiles with ai-edge-litert==2.2.0, so the
# shim comes from the v2.2.0 release. Bump all three together.
#
# The .so is gitignored (`**/jniLibs/**/*.so`) because runtime libraries are
# fetched, not committed -- see npubench/README.md for the same convention.
set -e
VERSION="${1:-v2.2.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../app/src/main/jniLibs/arm64-v8a"
URL="https://github.com/google-ai-edge/LiteRT/releases/download/$VERSION/litert_npu_runtime_libraries.zip"

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetch $URL"
curl -sSL -o "$TMP/libs.zip" "$URL"
unzip -q "$TMP/libs.zip" -d "$TMP"
cp "$TMP/google_tensor_runtime/src/main/jni/arm64-v8a/libLiteRtDispatch_GoogleTensor.so" "$DEST/"
echo "wrote $DEST/libLiteRtDispatch_GoogleTensor.so ($(stat -c%s "$DEST/libLiteRtDispatch_GoogleTensor.so") bytes)"
