package dev.pockettts

import java.io.Closeable
import java.io.File

/** Experimental GPU FlowLM path with persistent OpenCL packed K/V inputs. */
class GpuOpenClLmRunner(graph: File, highPriority: Boolean = false) : Closeable {
    private var handle = GpuOpenClLmBridge.open(graph.absolutePath, highPriority)

    val details: String get() = GpuOpenClLmBridge.details(requireHandle())
    val ready: Boolean get() = GpuOpenClLmBridge.ready(requireHandle())

    /** Voice arrays are group-major; the native runner seeds position-major OpenCL caches. */
    fun seed(k: FloatArray, v: FloatArray) = GpuOpenClLmBridge.seed(requireHandle(), k, v)

    fun step(
        emb: FloatArray, cos: FloatArray, sin: FloatArray, mask: FloatArray,
        noise: FloatArray, position: Int, fullOutput: Boolean = false,
    ): GpuOpenClLmStep {
        val timings = LongArray(6)
        val packed = GpuOpenClLmBridge.step(
            requireHandle(), emb, cos, sin, mask, noise, position, fullOutput, timings,
        )
        return GpuOpenClLmStep(packed, timings)
    }

    private fun requireHandle(): Long = handle.also { check(it != 0L) { "GPU runner is closed" } }

    override fun close() {
        if (handle != 0L) {
            GpuOpenClLmBridge.close(handle)
            handle = 0L
        }
    }
}

data class GpuOpenClLmStep(val packed: FloatArray, val timingsNs: LongArray)

internal object GpuOpenClLmBridge {
    init { System.loadLibrary("pockettts_gpu_opencl") }

    external fun open(path: String, highPriority: Boolean): Long
    external fun details(handle: Long): String
    external fun ready(handle: Long): Boolean
    external fun seed(handle: Long, k: FloatArray, v: FloatArray)
    external fun step(
        handle: Long, emb: FloatArray, cos: FloatArray, sin: FloatArray,
        mask: FloatArray, noise: FloatArray, position: Int, fullOutput: Boolean,
        timings: LongArray,
    ): FloatArray
    external fun close(handle: Long)
}

