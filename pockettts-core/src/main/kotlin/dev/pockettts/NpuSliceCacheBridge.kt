package dev.pockettts

import com.google.ai.edge.litert.TensorBuffer

/** Maps LiteRT 2.2.0 buffers and patches one position of the packed K/V cache. */
internal object NpuSliceCacheBridge {
    init { System.loadLibrary("pockettts_kv_slice") }

    /** LiteRtTensorBufferType for cache K/V followed by the packed step output. */
    external fun bufferTypes(
        cacheK: TensorBuffer, cacheV: TensorBuffer, stepOutput: TensorBuffer,
    ): IntArray

    /** Returns only eos + latent; timings are output-map, cache-map, copy, unmap ns. */
    external fun update(
        cacheK: TensorBuffer, cacheV: TensorBuffer, stepOutput: TensorBuffer,
        position: Int, capacity: Int, groups: Int, headDim: Int,
        positionMajor: Boolean,
        timings: LongArray,
    ): FloatArray

    /** Exact row check for opt-in diagnostics; returns the maximum absolute difference. */
    external fun rowMaxDifference(
        cacheK: TensorBuffer, cacheV: TensorBuffer, stepOutput: TensorBuffer,
        position: Int, capacity: Int, groups: Int, headDim: Int,
        positionMajor: Boolean,
    ): Float
}
