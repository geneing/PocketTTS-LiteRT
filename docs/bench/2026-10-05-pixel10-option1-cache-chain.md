# Option 1: Tensor G5 resident-cache first gate

Status on 2026-10-05: **host proof prepared; Pixel 10 gate pending**. No device latency, energy, buffer residency, or device-side copy claim is made here. The full stateful FlowLM export must wait for the device gate in the [research protocol](../flowlm-pixel10-optimization-research.md).

## Graph and environment

`scripts/probe_npu_cache_chain.py` exports a fixed-shape fp32 graph with cache `[2,96,512,64]` (25,165,824 bytes), one-row update `[2,96,1,64]` (49,152 bytes), and outputs updated cache plus a scalar from the updated row. The first row changes and the remaining cache passes through. Android's `FlowLmHarnessTest` has separate `probeNpuCacheChain`, `probeNpuCacheHostBaseline`, and `probeNpuCacheAlias` methods. Each runs one method in a foreground instrumentation process and writes `report.txt` under `Android/data/com.pockettts/files/flowlm-harness/cache-*/`.

| Item | Host result |
|---|---|
| Export toolchain | Existing WSL conversion environment, `litert-torch` |
| AOT toolchain | `ai-edge-litert==2.2.0`, Google Tensor SDK `2.2.0`, Tensor G5 target |
| Android runtime / dispatch | `litert:2.2.0`; repository's v2.2.0 Google Tensor dispatch shim (actual device load pending) |
| Source graph | `pt_npu_cache_chain.tflite`, 3,336 bytes, SHA-256 `2c71d52d06e6513865ed20d334b4ce0cd2e49c7d097b5b6ea1452e855822c74a` |
| AOT graph | `pt_npu_cache_chain_g5.tflite`, 363,376 bytes, SHA-256 `a5fb30d0c684c29b133f1e5cf23a80b82c318aa37ac058865b79f3b1b8f43282` |
| AOT partition | Subgraph 0 fully compiled; 9 of 9 ops offloaded to one Google Tensor G5 partition; 2.8 s host compilation |
| CPU tensor oracle | 3 of 3 steps; every output cache element and scalar exactly matched NumPy |
| Android harness build | `:app:compileDebugAndroidTestKotlin --offline --no-daemon` passed |

The two-bank harness writes the 25,165,824-byte cache once and a 49,152-byte row per step, reads one 4-byte scalar per step, then swaps the cache output into the next cache input. The host-write control stages the full 25,165,824-byte cache on every step using the same graph. The alias method is isolated because same-buffer input/output may be unsupported. The scalar read is the explicit completion point. The test records load, write, run, read/sync, and PSS separately.

The LiteRT 2.2.0 Kotlin `TensorBuffer` API exposes `read*` and `write*` but no actual buffer-type or AHardwareBuffer access method. The harness logs input/output supported buffer types, but **successful chaining alone does not prove device residency or absence of an internal full-cache copy**. Dispatch logs or a device trace are required to close this gate.

## Pixel 10 gate to run

Only one instrumentation method should run at a time. Install the APKs built from this option's worktree, push `scripts/out/pt_npu_cache_chain_g5.tflite` to `/sdcard/Android/data/com.pockettts/files/`, then run `com.pockettts.FlowLmHarnessTest#probeNpuCacheChain` and `#probeNpuCacheHostBaseline` separately. Run `#probeNpuCacheAlias` separately after the two-bank path. Capture `FlowLmHarness` and LiteRT/Google Tensor dispatch logcat around each run, device fingerprint, and each `report.txt`. Do not use `connectedDebugAndroidTest`: it removes the installed app and model directory.

| Required device observation | Result |
|---|---|
| Two-bank scalar sequence and cache reuse | Pending Pixel 10 |
| Supported and actual NPU buffer type | Pending Pixel 10; actual type unavailable through Kotlin API |
| Host cache bytes/step and staging latency | Pending Pixel 10; expected 0 recurring cache bytes for chain |
| NPU run and scalar read/sync latency | Pending Pixel 10 |
| Device-side full-cache copy, dispatch trace, peak PSS | Pending Pixel 10 |
| Same-buffer alias safety | Pending Pixel 10; two banks are the supported fallback |
| CPU int8 paired pipeline and power acceptance | Not applicable until the first gate passes and a full FlowLM variant exists |

Stop this branch before full FlowLM export if output-to-input binding fails or the runtime routes the 25 MiB cache through host memory. If chaining succeeds, inspect device-side copy cost and buffer type before claiming a performance win.
