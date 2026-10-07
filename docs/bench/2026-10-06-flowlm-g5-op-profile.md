# Pixel 10 FlowLM G5 runtime metrics probe

## Setup

The probe ran `FlowLmHarnessTest#npuResidentCacheSpeechPair` on Pixel 10
(`frankel`, Android 17) with the selective FFN12 static W8/A16 G5 graph,
position-major persistent cache, `g5HighPerformance=true`, Alba, seed 42, and
the short prompt `Hello there, how are you?`. The normal graph SHA-256 was
`183a71a2475e6fb68ab626788f8f2189a85d9fb661bcca8c8637e19ce2906d12`.

The app now brackets NPU inference with LiteRT's compiled-model hardware metrics
collection API and records the metric names and values in the harness report.
This probe uses the pinned LiteRT 2.2.0 C API because the Kotlin API does not
expose this collection path.

## Runtime results

The driver returned the same two metrics at detail levels 0 through 3:

| Detail level | Graph executions | `hardware_execution_time_us` | Other metric |
|---:|---:|---:|---|
| 0 | 26 | 249,807 | `number_of_graph_executions=26` |
| 1 | 26 | 252,603 | `number_of_graph_executions=26` |
| 2 | 26 | 249,342 | `number_of_graph_executions=26` |
| 3 | 26 | 253,956 | `number_of_graph_executions=26` |

That is about 9.6 ms of reported hardware time per graph execution. For the
detail-0 run, the end-to-end NPU synthesis took 889 ms versus 621 ms on the CPU
reference. The NPU FlowLM stage was 44 ms input preparation, 330 ms in the run
call, and 55 ms output read/cache update. These stage values include host-side
work; the vendor metric does not split them into operator or dispatch costs.

The harness reports 18 frames from each path, waveform correlation 0.809, and
SNR 4.61 dB for this short comparison. These are signal diagnostics, not an
intelligibility assessment.

## AOT timing diagnostic

LiteRT 2.2.0 accepts `google_tensor_dump_op_timings=true` in the Google Tensor
AOT compiler options ([pinned options header](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/c/options/litert_google_tensor_options.h#L68-L73)). The
FFN12 source graph compiled successfully with 694/694 operators in one
partition. Capturing the underlying compiler's combined output produced no
operator timing table or timing files. The timing-enabled graph had SHA-256
`955f10ffdb8c1a00302da57a0edd1043e4451f8f4e8e44647f8d7ea585ce4610`; it also
ran on the Pixel, where runtime collection still returned only the same two
aggregate metrics (`249,586 us` over 26 executions).

This confirms that the option changes the compiled output, but does not expose
operator timings through the compiler output or LiteRT metrics path exercised
here. The option's timing provenance and output format are not documented, so
the result is not treated as measured per-operator timing.

## Bottleneck findings and limits

The G5 model is one AOT partition. LiteRT's generic profiler can therefore see
the dispatch boundary, not the operators inside the compiled Tensor program.
The SouthBound interface also leaves metric names and detail-level behavior to
the vendor ([pinned interface](https://raw.githubusercontent.com/google-ai-edge/LiteRT/v2.2.0/litert/vendors/google_tensor/dispatch/sb_api.h)).
The Pixel 10 dispatch driver did not return per-operator identifiers or times
at levels 0-3.

The existing paired long-run measurements in
[`2026-10-06-g5-high-performance.md`](2026-10-06-g5-high-performance.md)
show the actionable host-visible costs: the FFN12 candidate's LM graph-run
stage averaged 14.351 s versus 15.014 s for CPU INT8, while the cache map and
row-update stages added about 2.16 s and the Mimi decoder transformer took
3.577 s versus 1.751 s in the CPU-LM arm. Total NPU synthesis still averaged
30.472 s versus 24.881 s on CPU. The current evidence points to cache movement
and decoder placement as end-to-end bottlenecks; it cannot rank internal NPU
operators.

The NPU FlowLM path remains opt-in. A true internal operator profile requires
Google Tensor SDK/runtime tooling that emits per-op measurements; public LiteRT
metrics on this device expose only aggregates. Compiling and timing separate
FlowLM blocks is a possible follow-up, but those numbers would be block-level
and would change compiler fusion and memory behavior.
