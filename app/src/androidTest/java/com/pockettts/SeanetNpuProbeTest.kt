package com.pockettts

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/** Device-side SEANet accelerator timing and output-dump probe. */
@RunWith(AndroidJUnit4::class)
class SeanetNpuProbeTest {

    private data class TensorSpec(
        val shape: IntArray,
        val elements: Int,
        val type: Int,
        val scale: Float = 0f,
        val zeroPoint: Int = 0,
    )

    private data class ModelSpec(val inputs: List<TensorSpec>, val outputs: List<TensorSpec>)

    @Test
    fun probe() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = ctx.getExternalFilesDir(null) ?: ctx.filesDir
        val npuEnv = Environment.create(
            ctx,
            mapOf(Environment.Option.DispatchLibraryDir to ctx.applicationInfo.nativeLibraryDir),
        )
        val inputFile = File(dir, "seanet_input.bin")
        val input = if (inputFile.isFile) {
            val bytes = inputFile.readBytes()
            FloatArray(bytes.size / Float.SIZE_BYTES).also {
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it)
            }
        } else {
            Log.i(TAG, "skip input: no seanet_input.bin")
            return
        }

        try {
            check(input.size == 512 * 512) { "expected feat[1,512,512], got ${input.size} floats" }
            val nearGpu = File(dir, "seanet_near_w512_0-2_padinsert_fp16.tflite")
            val nearNpu = File(dir, "seanet_near_w512_0-2_padinsert_fp16_aot.tflite")
            val farGpu = File(dir, "seanet_far_w512_3-26_padinsert_fp16.tflite")
            val farNpu = File(dir, "seanet_far_w512_3-26_padinsert_fp16_aot.tflite")
            val simpleGpu = File(dir, "seanet_near_w512_0-0_padinsert_fp16.tflite")
            val simpleNpu = File(dir, "seanet_near_w512_0-0_padinsert_fp16_aot.tflite")
            val nearHalf = File(dir, "seanet_near_w512_0-2_padinsert_fp16_aot_half.tflite")
            val nearNoTrunc = File(dir, "seanet_near_w512_0-2_padinsert_fp16_aot_no_truncation.tflite")
            val farHalf = File(dir, "seanet_far_w512_3-26_padinsert_fp16_aot_half.tflite")
            val farNoTrunc = File(dir, "seanet_far_w512_3-26_padinsert_fp16_aot_no_truncation.tflite")
            val nearInt8Gpu = File(dir, "seanet_near_w512_0-2_padinsert_int8.tflite")
            val nearInt8Npu = File(dir, "seanet_near_w512_0-2_padinsert_int8_aot.tflite")
            val farInt8Gpu = File(dir, "seanet_far_w512_3-26_padinsert_int8.tflite")
            val farInt8Npu = File(dir, "seanet_far_w512_3-26_padinsert_int8_aot.tflite")

            val nearGpuOut = runIfPresent(nearGpu, Accelerator.GPU, npuEnv, input,
                "seanet_near_gpu.bin")
            val nearNpuOut = runIfPresent(nearNpu, Accelerator.NPU, npuEnv, input,
                "seanet_near_npu.bin")
            if (nearGpuOut != null) {
                Log.i(TAG, "SHAPE-MATCH near pair: GPU and NPU both receive feat[1,512,512]")
                val farGpuOut = runIfPresent(farGpu, Accelerator.GPU, npuEnv, nearGpuOut,
                    "seanet_far_gpu_same_near_gpu_input.bin")
                val farNpuOut = runIfPresent(farNpu, Accelerator.NPU, npuEnv, nearGpuOut,
                    "seanet_far_npu_same_near_gpu_input.bin")
                if (farGpuOut != null && farNpuOut != null) {
                    Log.i(TAG, "SHAPE-MATCH far pair: GPU and NPU both receive the same near-GPU [1,256,3072] output")
                }
                runIfPresent(farHalf, Accelerator.NPU, npuEnv, nearGpuOut,
                    "seanet_far_npu_half_same_input.bin")
                runIfPresent(farNoTrunc, Accelerator.NPU, npuEnv, nearGpuOut,
                    "seanet_far_npu_no_truncation_same_input.bin")
            }
            runIfPresent(nearHalf, Accelerator.NPU, npuEnv, input,
                "seanet_near_npu_half.bin")
            runIfPresent(nearNoTrunc, Accelerator.NPU, npuEnv, input,
                "seanet_near_npu_no_truncation.bin")

            val nearInt8GpuOut = runIfPresent(nearInt8Gpu, Accelerator.GPU, npuEnv, input,
                "seanet_near_int8_gpu.bin")
            val nearInt8NpuOut = runIfPresent(nearInt8Npu, Accelerator.NPU, npuEnv, input,
                "seanet_near_int8_npu.bin")
            if (nearInt8GpuOut != null) {
                runIfPresent(farInt8Gpu, Accelerator.GPU, npuEnv, nearInt8GpuOut,
                    "seanet_far_int8_gpu_same_input.bin")
                runIfPresent(farInt8Npu, Accelerator.NPU, npuEnv, nearInt8GpuOut,
                    "seanet_far_int8_npu_same_input.bin")
            }
            if (nearInt8NpuOut != null) {
                try {
                    runPipeline(nearInt8Npu, farInt8Npu, Accelerator.NPU, Accelerator.NPU,
                        npuEnv, input, "seanet_pipeline_int8_npu.bin")
                } catch (t: Throwable) {
                    Log.i(TAG, "FAIL int8 pipeline NPU ${t.javaClass.simpleName}: ${t.message}")
                }
                try {
                    runPipeline(nearInt8Npu, farInt8Gpu, Accelerator.NPU, Accelerator.GPU,
                        npuEnv, input, "seanet_pipeline_int8_npu_gpu.bin")
                } catch (t: Throwable) {
                    Log.i(TAG, "FAIL int8 pipeline NPU->GPU ${t.javaClass.simpleName}: ${t.message}")
                }
            }
            if (nearInt8Gpu.isFile && farInt8Gpu.isFile) {
                try {
                    runPipeline(nearInt8Gpu, farInt8Gpu, Accelerator.GPU, Accelerator.GPU,
                        npuEnv, input, "seanet_pipeline_int8_gpu.bin")
                } catch (t: Throwable) {
                    Log.i(TAG, "FAIL int8 pipeline GPU ${t.javaClass.simpleName}: ${t.message}")
                }
            }
            if (nearGpuOut != null && nearNpuOut != null) {
                try {
                    runPipeline(nearGpu, farGpu, Accelerator.GPU, Accelerator.GPU, npuEnv, input,
                        "seanet_pipeline_gpu.bin")
                } catch (t: Throwable) {
                    Log.i(TAG, "FAIL pipeline GPU ${t.javaClass.simpleName}: ${t.message}")
                }
                try {
                    runPipeline(nearNpu, farNpu, Accelerator.NPU, Accelerator.NPU, npuEnv, input,
                        "seanet_pipeline_npu.bin")
                } catch (t: Throwable) {
                    Log.i(TAG, "FAIL pipeline NPU ${t.javaClass.simpleName}: ${t.message}")
                }
                try {
                    runPipeline(nearNpu, farGpu, Accelerator.NPU, Accelerator.GPU, npuEnv, input,
                        "seanet_pipeline_npu_gpu.bin")
                } catch (t: Throwable) {
                    Log.i(TAG, "FAIL pipeline NPU->GPU ${t.javaClass.simpleName}: ${t.message}")
                }
                try {
                    runPipeline(nearGpu, farNpu, Accelerator.GPU, Accelerator.NPU, npuEnv, input,
                        "seanet_pipeline_gpu_npu.bin")
                } catch (t: Throwable) {
                    Log.i(TAG, "FAIL pipeline GPU->NPU ${t.javaClass.simpleName}: ${t.message}")
                }
            }

            runIfPresent(simpleGpu, Accelerator.GPU, npuEnv, input,
                "seanet_simple_near_gpu.bin")
            runIfPresent(simpleNpu, Accelerator.NPU, npuEnv, input,
                "seanet_simple_near_npu.bin")

            runIfPresent(File(dir, "pt_mimi_deconly_w512_fp16.tflite"),
                Accelerator.GPU, npuEnv, input, "seanet_full_gpu_gold.bin")
        } finally {
            npuEnv.close()
        }
        Log.i("Probe", "done")
    }

    private fun runIfPresent(
        file: File,
        accelerator: Accelerator,
        npuEnv: Environment,
        input: FloatArray,
        outputName: String,
    ): FloatArray? {
        if (!file.isFile) {
            Log.i(TAG, "skip ${file.name} (missing)")
            return null
        }
        return try {
            runGraph(file, accelerator, npuEnv, input, outputName)
        } catch (t: Throwable) {
            Log.i(TAG, "FAIL ${file.name} accel=$accelerator ${t.javaClass.simpleName}: ${t.message}")
            null
        }
    }

    private fun runGraph(
        file: File,
        accelerator: Accelerator,
        npuEnv: Environment,
        input: FloatArray,
        outputName: String = file.name.removeSuffix(".tflite") + ".bin",
    ): FloatArray {
        val metadata = readModelSpec(file)

        val startCreate = System.nanoTime()
        val model = CompiledModel.create(
            file.absolutePath,
            CompiledModel.Options(accelerator),
            if (accelerator == Accelerator.NPU) npuEnv else null,
        )
        try {
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            check(inputs.size == 1 && outputs.size == 1) {
                "expected one input/output, got ${inputs.size}/${outputs.size}"
            }
            val inputSpec = metadata.inputs.firstOrNull()
                ?: error("model has no input tensor: ${file.name}")
            val inputShape = inputSpec.shape
            val inputCount = inputSpec.elements
            val outputShape = metadata.outputs.firstOrNull()?.shape
            check(inputCount == input.size) {
                "input size mismatch: graph wants $inputCount, dump has ${input.size}"
            }
            writeInput(inputs[0], input, inputSpec)
            Log.i(
                TAG,
                "graph=${file.name} accel=$accelerator " +
                    "inputShape=${inputShape.contentToString()} inputFloats=$inputCount " +
                    "inputType=${inputSpec.type} outputShape=${outputShape?.contentToString() ?: "unknown"} " +
                    "outputType=${metadata.outputs.firstOrNull()?.type}",
            )

            val firstRunStart = System.nanoTime()
            model.run(inputs, outputs)
            val coldMs = (System.nanoTime() - startCreate) / 1e6
            val firstRunMs = (System.nanoTime() - firstRunStart) / 1e6
            model.run(inputs, outputs) // warmup

            val times = DoubleArray(TIMED_RUNS)
            var lastOutput = FloatArray(0)
            for (i in times.indices) {
                val t = System.nanoTime()
                model.run(inputs, outputs)
                times[i] = (System.nanoTime() - t) / 1e6
                if (i == times.lastIndex) {
                    lastOutput = readOutput(outputs[0], metadata.outputs.first())
                }
            }
            val sorted = times.sorted()
            val avg = times.average()
            val p50 = if (sorted.size % 2 == 1) {
                sorted[sorted.size / 2]
            } else {
                (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2.0
            }
            Log.i(
                TAG,
                "RESULT ${file.name} accel=$accelerator runs=${times.size} " +
                    "coldMs=${fmt(coldMs)} firstRunMs=${fmt(firstRunMs)} avgMs=${fmt(avg)} " +
                    "minMs=${fmt(sorted.first())} p50Ms=${fmt(p50)} outputFloats=${lastOutput.size}",
            )
            File(file.parentFile, outputName).writeBytes(toLittleEndian(lastOutput))

            measurePower(file.name, inputs, outputs, model)
            return lastOutput
        } finally {
            model.close()
        }
    }

    private fun runPipeline(
        nearFile: File,
        farFile: File,
        nearAccelerator: Accelerator,
        farAccelerator: Accelerator,
        npuEnv: Environment,
        input: FloatArray,
        outputName: String,
    ) {
        if (!nearFile.isFile || !farFile.isFile) {
            Log.i(TAG, "skip pipeline $nearAccelerator->$farAccelerator " +
                "(missing ${nearFile.name} or ${farFile.name})")
            return
        }
        val near = CompiledModel.create(
            nearFile.absolutePath,
            CompiledModel.Options(nearAccelerator),
            if (nearAccelerator == Accelerator.NPU) npuEnv else null,
        )
        val far = try {
            CompiledModel.create(
                farFile.absolutePath,
                CompiledModel.Options(farAccelerator),
                if (farAccelerator == Accelerator.NPU) npuEnv else null,
            )
        } catch (t: Throwable) {
            near.close()
            throw t
        }
        try {
            val nearSpec = readModelSpec(nearFile)
            val farSpec = readModelSpec(farFile)
            val nearInputs = near.createInputBuffers()
            val nearOutputs = near.createOutputBuffers()
            val farInputs = far.createInputBuffers()
            val farOutputs = far.createOutputBuffers()
            check(nearInputs.size == 1 && nearOutputs.size == 1 &&
                farInputs.size == 1 && farOutputs.size == 1) { "pipeline expects single-input/output graphs" }
            writeInput(nearInputs[0], input, nearSpec.inputs.first())
            Log.i(TAG, "pipeline near=$nearAccelerator far=$farAccelerator " +
                "${nearFile.name} feat[1,512,512] " +
                "-> [1,256,3072] -> far=${farFile.name} -> [1,1,61440]")
            near.run(nearInputs, nearOutputs)
            bridge(nearOutputs[0], nearSpec.outputs.first(), farInputs[0], farSpec.inputs.first())
            far.run(farInputs, farOutputs)
            near.run(nearInputs, nearOutputs)
            bridge(nearOutputs[0], nearSpec.outputs.first(), farInputs[0], farSpec.inputs.first())
            far.run(farInputs, farOutputs)

            val times = DoubleArray(TIMED_RUNS)
            var lastOutput = FloatArray(0)
            for (i in times.indices) {
                val start = System.nanoTime()
                near.run(nearInputs, nearOutputs)
                bridge(nearOutputs[0], nearSpec.outputs.first(), farInputs[0], farSpec.inputs.first())
                far.run(farInputs, farOutputs)
                times[i] = (System.nanoTime() - start) / 1e6
                if (i == times.lastIndex) {
                    lastOutput = readOutput(farOutputs[0], farSpec.outputs.first())
                }
            }
            val sorted = times.sorted()
            val p50 = if (sorted.size % 2 == 1) sorted[sorted.size / 2]
                else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2.0
            Log.i(TAG, "PIPELINE near=$nearAccelerator far=$farAccelerator " +
                "runs=${times.size} avgMs=${fmt(times.average())} " +
                "minMs=${fmt(sorted.first())} p50Ms=${fmt(p50)} outputFloats=${lastOutput.size}")
            File(nearFile.parentFile, outputName).writeBytes(toLittleEndian(lastOutput))
        } finally {
            far.close()
            near.close()
        }
    }

    private fun measurePower(
        name: String,
        inputs: List<com.google.ai.edge.litert.TensorBuffer>,
        outputs: List<com.google.ai.edge.litert.TensorBuffer>,
        model: CompiledModel,
    ) {
        val currentPath = "/sys/class/power_supply/battery/current_now"
        val voltagePath = "/sys/class/power_supply/battery/voltage_now"
        val before = readPower(currentPath, voltagePath)
        if (before == null) {
            Log.i(TAG, "POWER $name unavailable: battery current_now/voltage_now are not readable")
            return
        }
        val start = System.nanoTime()
        repeat(POWER_RUNS) { model.run(inputs, outputs) }
        val elapsedSeconds = (System.nanoTime() - start) / 1e9
        val after = readPower(currentPath, voltagePath)
        if (after == null) {
            Log.i(TAG, "POWER $name unavailable: could not read battery current_now/voltage_now after loop")
            return
        }
        val meanPowerW = (abs(before.first * before.second) + abs(after.first * after.second)) / 2e12
        val joulesPerRun = meanPowerW * elapsedSeconds / POWER_RUNS
        Log.i(
            TAG,
            "POWER $name runs=$POWER_RUNS elapsedSec=${fmt(elapsedSeconds)} " +
                "currentUa=${before.first}->${after.first} voltageUv=${before.second}->${after.second} " +
                "endpointEstimateJPerRun=${"%.6g".format(Locale.US, joulesPerRun)}",
        )
    }

    private fun readPower(currentPath: String, voltagePath: String): Pair<Long, Long>? = try {
        File(currentPath).readText().trim().toLong() to
            File(voltagePath).readText().trim().toLong()
    } catch (_: Throwable) {
        null
    }

    private fun readModelSpec(file: File): ModelSpec {
        val model = try {
            Interpreter(file).use { interpreter ->
                fun tensorSpec(tensor: org.tensorflow.lite.Tensor): TensorSpec {
                    val qp = tensor.quantizationParams()
                    return TensorSpec(tensor.shape(), tensor.numElements(), tensor.dataType().ordinal,
                        qp.scale, qp.zeroPoint)
                }
                ModelSpec(
                    (0 until interpreter.inputTensorCount).map { tensorSpec(interpreter.getInputTensor(it)) },
                    (0 until interpreter.outputTensorCount).map { tensorSpec(interpreter.getOutputTensor(it)) },
                )
            }
        } catch (t: Throwable) {
            Log.i(TAG, "interpreter shape query unavailable for ${file.name}: ${t.message}")
            readFlatbufferSpec(file)
        }
        if ((model.inputs + model.outputs).none { it.type == TFLITE_INT8 && it.scale == 0f }) {
            return model
        }

        val source = if (file.name.endsWith("_aot.tflite")) {
            File(file.parentFile, file.name.removeSuffix("_aot.tflite") + ".tflite")
        } else file
        if (!source.isFile) return model
        return try {
            Interpreter(source).use { interpreter ->
                fun tensorSpecs(tensors: List<org.tensorflow.lite.Tensor>, fallback: List<TensorSpec>) =
                    tensors.mapIndexed { i, tensor ->
                        val old = fallback[i]
                        val qp = tensor.quantizationParams()
                        old.copy(scale = qp.scale, zeroPoint = qp.zeroPoint)
                    }
                ModelSpec(
                    tensorSpecs((0 until interpreter.inputTensorCount).map { interpreter.getInputTensor(it) },
                        model.inputs),
                    tensorSpecs((0 until interpreter.outputTensorCount).map { interpreter.getOutputTensor(it) },
                        model.outputs),
                )
            }
        } catch (t: Throwable) {
            throw IllegalStateException("could not read quantization params for ${file.name}", t)
        }
    }

    private fun writeInput(
        buffer: com.google.ai.edge.litert.TensorBuffer,
        values: FloatArray,
        spec: TensorSpec,
    ) {
        when (spec.type) {
            TFLITE_FLOAT32 -> buffer.writeFloat(values)
            TFLITE_INT8 -> {
                check(spec.scale > 0f) { "missing input quantization scale" }
                val quantized = ByteArray(values.size) { i ->
                    (kotlin.math.round(values[i] / spec.scale).toInt() + spec.zeroPoint)
                        .coerceIn(-128, 127).toByte()
                }
                buffer.writeInt8(quantized)
            }
            else -> error("unsupported input tensor type ${spec.type}")
        }
    }

    private fun bridge(
        source: com.google.ai.edge.litert.TensorBuffer,
        sourceSpec: TensorSpec,
        target: com.google.ai.edge.litert.TensorBuffer,
        targetSpec: TensorSpec,
    ) {
        val scalesMatch = kotlin.math.abs(sourceSpec.scale - targetSpec.scale) <=
            maxOf(sourceSpec.scale, targetSpec.scale) * 1e-5f
        if (sourceSpec.type == TFLITE_INT8 && targetSpec.type == TFLITE_INT8 &&
            sourceSpec.zeroPoint == targetSpec.zeroPoint && scalesMatch) {
            target.writeInt8(source.readInt8())
        } else {
            writeInput(target, readOutput(source, sourceSpec), targetSpec)
        }
    }

    private fun readOutput(
        buffer: com.google.ai.edge.litert.TensorBuffer,
        spec: TensorSpec,
    ): FloatArray = when (spec.type) {
        TFLITE_FLOAT32 -> buffer.readFloat()
        TFLITE_INT8 -> {
            check(spec.scale > 0f) { "missing output quantization scale" }
            buffer.readInt8().map {
                (it.toInt() - spec.zeroPoint) * spec.scale
            }.toFloatArray()
        }
        else -> error("unsupported output tensor type ${spec.type}")
    }

    /** Reads shape/type metadata directly when an AOT DISPATCH_OP blocks TFLite allocation. */
    private fun readFlatbufferSpec(file: File): ModelSpec {
        val b = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        fun field(table: Int, index: Int): Int {
            val vtable = table - b.getInt(table)
            val vtableSize = b.getShort(vtable).toInt() and 0xffff
            val entry = vtable + 4 + index * 2
            if (entry + 2 > vtable + vtableSize) return 0
            val offset = b.getShort(entry).toInt() and 0xffff
            return if (offset == 0) 0 else table + offset
        }
        fun vector(table: Int, index: Int): Int {
            val f = field(table, index)
            return if (f == 0) 0 else f + b.getInt(f)
        }
        fun vectorInts(table: Int, index: Int): List<Int> {
            val v = vector(table, index)
            if (v == 0) return emptyList()
            val n = b.getInt(v)
            return (0 until n).map { b.getInt(v + 4 + it * 4) }
        }
        fun vectorTable(table: Int, index: Int, item: Int): Int {
            val v = vector(table, index)
            val slot = v + 4 + item * 4
            return slot + b.getInt(slot)
        }
        fun tensorSpec(subgraph: Int, tensorIndex: Int): TensorSpec {
            val tensor = vectorTable(subgraph, 0, tensorIndex)
            val shape = vectorInts(tensor, 0).toIntArray()
            val typeField = field(tensor, 1)
            val flatbufferType = if (typeField == 0) 0 else b.get(typeField).toInt() and 0xff
            val type = if (flatbufferType == TFLITE_FLATBUFFER_INT8) TFLITE_INT8 else flatbufferType
            return TensorSpec(shape, shape.fold(1) { n, d -> n * d }, type)
        }

        val model = b.getInt(0)
        val subgraph = vectorTable(model, 2, 0)
        val inputIndices = vectorInts(subgraph, 1)
        val outputIndices = vectorInts(subgraph, 2)
        check(inputIndices.isNotEmpty() && outputIndices.isNotEmpty()) {
            "flatbuffer has no input/output tensors"
        }
        return ModelSpec(
            inputIndices.map { tensorSpec(subgraph, it) },
            outputIndices.map { tensorSpec(subgraph, it) },
        )
    }

    private fun toLittleEndian(values: FloatArray): ByteArray {
        val bytes = ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(bytes::putFloat)
        return bytes.array()
    }

    private fun fmt(value: Double): String = String.format(Locale.US, "%.3f", value)

    companion object {
        private const val TAG = "SeanetNpuProbe"
        private const val TFLITE_FLOAT32 = 0
        private const val TFLITE_INT8 = 7
        private const val TFLITE_FLATBUFFER_INT8 = 9
        private const val TIMED_RUNS = 20
        private const val POWER_RUNS = 200
    }
}
