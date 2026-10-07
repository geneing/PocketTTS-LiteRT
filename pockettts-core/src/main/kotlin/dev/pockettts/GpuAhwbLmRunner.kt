package dev.pockettts

import java.io.Closeable
import java.io.File

/** Experimental GPU FlowLM path with persistent AHardwareBuffer K/V inputs. */
class GpuAhwbLmRunner(graph: File) : Closeable {
    private var handle = GpuAhwbLmBridge.open(graph.absolutePath)

    val details: String get() = GpuAhwbLmBridge.details(requireHandle())
    val ready: Boolean get() = GpuAhwbLmBridge.ready(requireHandle())

    /** Voice arrays are group-major; the native runner seeds position-major AHWB caches. */
    fun seed(k: FloatArray, v: FloatArray) = GpuAhwbLmBridge.seed(requireHandle(), k, v)

    fun step(
        emb: FloatArray, cos: FloatArray, sin: FloatArray, mask: FloatArray,
        noise: FloatArray, position: Int,
    ): GpuAhwbLmStep {
        val timings = LongArray(6)
        val packed = GpuAhwbLmBridge.step(
            requireHandle(), emb, cos, sin, mask, noise, position, timings,
        )
        return GpuAhwbLmStep(packed, timings)
    }

    private fun requireHandle(): Long = handle.also { check(it != 0L) { "GPU runner is closed" } }

    override fun close() {
        if (handle != 0L) {
            GpuAhwbLmBridge.close(handle)
            handle = 0L
        }
    }
}

data class GpuAhwbLmStep(val packed: FloatArray, val timingsNs: LongArray)

internal object GpuAhwbLmBridge {
    init { System.loadLibrary("pockettts_gpu_ahwb") }

    external fun open(path: String): Long
    external fun details(handle: Long): String
    external fun ready(handle: Long): Boolean
    external fun seed(handle: Long, k: FloatArray, v: FloatArray)
    external fun step(
        handle: Long, emb: FloatArray, cos: FloatArray, sin: FloatArray,
        mask: FloatArray, noise: FloatArray, position: Int, timings: LongArray,
    ): FloatArray
    external fun close(handle: Long)
}
