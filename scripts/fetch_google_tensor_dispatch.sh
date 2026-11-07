#!/bin/bash
# The Google Tensor NPU dispatch shim is no longer fetched by hand.
#
# LiteRT 2.3.0 split vendor runtimes into separate Maven packages, and
# `com.google.ai.edge.litert:litert-npu-runtime-google-tensor:2.3.2` ships
# `libLiteRtDispatch_GoogleTensor.so` (plus the on-device compiler plugin) in
# its AAR. `pockettts-core` depends on it, so Gradle packages the shim into
# `lib/arm64-v8a/` and `PocketTtsEngine.npuEnv()` finds it through
# `Environment.Option.DispatchLibraryDir`.
#
# The compiler plugin, Android runtime and dispatch shim must all be the same
# LiteRT release: this app pins 2.3.2 for runtime and dispatch.
#
# This script is kept as a no-op so older reproduction steps still succeed.
set -e
echo "libLiteRtDispatch_GoogleTensor.so is provided by"
echo "  com.google.ai.edge.litert:litert-npu-runtime-google-tensor:2.3.2"
echo "no manual download is needed"
