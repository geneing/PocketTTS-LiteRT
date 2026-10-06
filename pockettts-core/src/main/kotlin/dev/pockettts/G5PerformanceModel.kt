package dev.pockettts

import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable

/** Opt-in LiteRT 2.2.0 C-API probe for Google Tensor HIGH_PERFORMANCE mode. */
class G5PerformanceModel private constructor(private var nativeHandle: Long) : Closeable {
    fun run(inputs: List<TensorBuffer>, outputs: List<TensorBuffer>, signatureIndex: Int) {
        check(nativeHandle != 0L) { "G5 performance model is closed" }
        nativeRun(nativeHandle, inputs.toTypedArray(), outputs.toTypedArray(), signatureIndex)
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeClose(nativeHandle)
            nativeHandle = 0L
        }
    }

    private external fun nativeRun(
        handle: Long, inputs: Array<TensorBuffer>, outputs: Array<TensorBuffer>, signatureIndex: Int,
    )
    private external fun nativeClose(handle: Long)

    companion object {
        init { System.loadLibrary("pockettts_kv_slice") }

        /** The [environment] must outlive this model. */
        fun create(environment: Environment, modelPath: String): G5PerformanceModel =
            G5PerformanceModel(nativeCreate(environment, modelPath))

        private external fun nativeCreate(environment: Environment, modelPath: String): Long
    }
}
