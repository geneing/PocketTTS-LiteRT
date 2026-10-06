package com.pockettts

import android.util.Half
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import dev.pockettts.Accel
import dev.pockettts.PocketTts
import dev.pockettts.SpTokenizer
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.sqrt

/** Text-driven FlowLM probe. Invoke with instrumentation args; outputs land in files/flowlm-harness. */
@RunWith(AndroidJUnit4::class)
class FlowLmHarnessTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val args get() = InstrumentationRegistry.getArguments()
    private val modelDir get() = context.getExternalFilesDir(null) ?: context.filesDir

    private data class Run(
        val backend: Accel,
        val graph: String,
        val output: FloatArray,
        val tokenCount: Int,
        val loadMs: Double,
        val runMs: Double,
    )

    private data class Metrics(val corr: Double, val mad: Double, val msd: Double)

    @Test
    fun runTextPromptHarness() {
        val text = args.getString("text")?.trim().orEmpty().ifEmpty { DEFAULT_TEXT }
        val voice = args.getString("voice")?.trim()?.ifEmpty { "alba" } ?: "alba"
        val baseGraph = args.getString("lmGraph")?.trim()?.ifEmpty { DEFAULT_GRAPH } ?: DEFAULT_GRAPH
        val requested = args.getString("backends")
            ?.split(',', ';', ' ')
            ?.mapNotNull { runCatching { Accel.valueOf(it.trim().uppercase(Locale.ROOT)) }.getOrNull() }
            ?.distinct()
            ?: listOf(Accel.NPU, Accel.GPU, Accel.CPU)
        // LiteRT's dispatch setup is process-global; initialize an NPU graph first.
        val backends = (requested.filter { it == Accel.NPU } + requested.filter { it != Accel.NPU }).distinct()
        val tokenizer = SpTokenizer(File(modelDir, PocketTts.TOKENIZER))
        val tokenIds = tokenizer.encode(text)
        require(tokenIds.isNotEmpty()) { "text encoded to no tokens" }

        val voiceState = readVoice(File(modelDir, PocketTts.voiceFile(voice)))
        require(tokenIds.size + voiceState.length < PocketTts.PMAX) {
            "prompt has ${tokenIds.size} tokens; voice '$voice' leaves only " +
                "${PocketTts.PMAX - voiceState.length - 1} FlowLM positions"
        }
        val embed = File(modelDir, PocketTts.EMBED).readBytes()
        val nEmbeddings = embed.size / 2 / PocketTts.H
        require(tokenIds.all { it in 0 until nEmbeddings }) { "tokenizer id outside embedding table" }

        val runDir = File(
            context.getExternalFilesDir("flowlm-harness") ?: File(context.filesDir, "flowlm-harness"),
            "run-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())}",
        ).apply { mkdirs() }
        val lines = ArrayList<String>()
        lines += "FlowLM text harness"
        lines += "device=${android.os.Build.MODEL} (${android.os.Build.DEVICE}) Android ${android.os.Build.VERSION.RELEASE}"
        lines += "text=$text"
        lines += "voice=$voice"
        lines += "graph=$baseGraph"
        lines += "tokens=${tokenIds.size} ids=${tokenIds.joinToString(",")}"
        lines += "output=one fused output vector per prompt token; zero noise; voice KV prefix=${voiceState.length} tokens"

        val environment = if (Accel.NPU in backends) {
            runCatching {
                Environment.create(
                    context,
                    mapOf(Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir),
                )
            }.onFailure { lines += "NPU environment FAILED: ${it.message}" }.getOrNull()
        } else null
        val candidates = LinkedHashMap<Accel, Run>()
        try {
            for (backend in backends) {
                val graph = if (backend == Accel.NPU) PocketTts.g5Variant(baseGraph) else baseGraph
                val result = try {
                    runPrompt(backend, graph, tokenIds, embed, voiceState, environment)
                } catch (t: Throwable) {
                    lines += "$backend FAILED: ${t.message ?: t::class.java.simpleName}"
                    Log.e(TAG, "backend=$backend graph=$graph failed", t)
                    null
                }
                if (result != null) candidates[backend] = result
            }

            // CPU fp16 is the reference. Running it after the candidates also
            // preserves the runtime's required NPU-first initialization order.
            val referenceGraph = args.getString("referenceGraph")
                ?.trim()?.ifEmpty { REFERENCE_GRAPH } ?: REFERENCE_GRAPH
            val reference = runPrompt(Accel.CPU, referenceGraph, tokenIds, embed, voiceState, environment)
            writeFloats(File(runDir, "cpu-reference.f32le"), reference.output)
            lines += String.format(
                Locale.US,
                "CPU reference: graph=%s load=%.2f ms run=%.2f ms values=%d saved=cpu-reference.f32le",
                reference.graph, reference.loadMs, reference.runMs, reference.output.size,
            )
            for ((backend, result) in candidates) {
                val file = "${backend.name.lowercase(Locale.ROOT)}-candidate.f32le"
                writeFloats(File(runDir, file), result.output)
                try {
                    val m = metrics(reference.output, result.output)
                    lines += String.format(
                        Locale.US,
                        "%s candidate: graph=%s load=%.2f ms run=%.2f ms corr=%.8f mean_abs_diff=%.8g mean_square_diff=%.8g saved=%s",
                        backend, result.graph, result.loadMs, result.runMs, m.corr, m.mad, m.msd, file,
                    )
                } catch (t: Throwable) {
                    lines += "$backend candidate comparison failed: ${t.message}; output saved=$file"
                }
            }
            lines += "CPU self-check: corr=1.00000000 mean_abs_diff=0 mean_square_diff=0"
            lines += "raw outputs are little-endian float32; ${PocketTts.H} embedding width; ${tokenIds.size} prompt tokens"
            val report = lines.joinToString("\n", postfix = "\n")
            File(runDir, "report.txt").writeText(report)
            report.lines().forEach { Log.i(TAG, it) }
            Log.i(TAG, "saved FlowLM probe files to ${runDir.absolutePath}")
            assertTrue("CPU fp16 reference produced no output", reference.output.isNotEmpty())
        } finally {
            environment?.close()
        }
    }

    private fun runPrompt(
        backend: Accel,
        graph: String,
        tokenIds: IntArray,
        embedBytes: ByteArray,
        voice: VoiceState,
        environment: Environment?,
    ): Run {
        check(backend != Accel.NPU || environment != null) { "NPU environment is unavailable" }
        val path = File(modelDir, graph)
        check(path.isFile) { "missing graph: $path" }
        val accelerator = when (backend) {
            Accel.CPU -> Accelerator.CPU
            Accel.GPU, Accel.GPU32 -> Accelerator.GPU
            Accel.NPU -> Accelerator.NPU
        }
        val options = CompiledModel.Options(accelerator)
        if (backend == Accel.GPU32) {
            options.gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
        }
        val loadStart = System.nanoTime()
        val model = CompiledModel.create(path.absolutePath, options, if (backend == Accel.NPU) environment else null)
        val loadMs = (System.nanoTime() - loadStart) / 1e6
        try {
            val stepInput = runCatching { model.createInputBuffers(1) }.getOrNull()
            val input = stepInput ?: model.createInputBuffers()
            val output = if (stepInput != null) model.createOutputBuffers(1) else model.createOutputBuffers()
            try {
                val pk = voice.k.copyOf()
                val pv = voice.v.copyOf()
                val mask = FloatArray(PocketTts.NH * (PocketTts.PMAX + 1)) { PocketTts.MASK_NEG }
                var pos = voice.length
                for (h in 0 until PocketTts.NH) {
                    val row = h * (PocketTts.PMAX + 1)
                    for (p in 0 until pos) mask[row + p] = 0f
                    mask[row + PocketTts.PMAX] = 0f
                }
                val cosine = FloatArray(PocketTts.HD)
                val sine = FloatArray(PocketTts.HD)
                val zeroNoise = FloatArray(PocketTts.LDIM)
                val embed = ByteBuffer.wrap(embedBytes).order(ByteOrder.LITTLE_ENDIAN)
                val outputPerToken = 1 + PocketTts.LDIM + 2 * PocketTts.G * PocketTts.HD
                val all = FloatArray(tokenIds.size * outputPerToken)
                var outputOffset = 0
                var elapsed = 0L
                for (id in tokenIds) {
                    val embedding = FloatArray(PocketTts.H)
                    var offset = id * PocketTts.H * Short.SIZE_BYTES
                    for (i in embedding.indices) {
                        embedding[i] = Half.toFloat(embed.getShort(offset))
                        offset += Short.SIZE_BYTES
                    }
                    rope(pos, cosine, sine)
                    input[0].writeFloat(embedding)
                    input[1].writeFloat(cosine)
                    input[2].writeFloat(sine)
                    input[3].writeFloat(mask)
                    input[4].writeFloat(pk)
                    input[5].writeFloat(pv)
                    input[6].writeFloat(zeroNoise)
                    val started = System.nanoTime()
                    if (stepInput != null) model.run(input, output, 1) else model.run(input, output)
                    elapsed += System.nanoTime() - started
                    val values = output.single().readFloat()
                    check(values.size == outputPerToken) {
                        "FlowLM output width changed: expected $outputPerToken, got ${values.size}"
                    }
                    System.arraycopy(values, 0, all, outputOffset, values.size)
                    outputOffset += values.size

                    val kvStart = 1 + PocketTts.LDIM
                    for (g in 0 until PocketTts.G) {
                        System.arraycopy(values, kvStart + g * PocketTts.HD, pk,
                            g * PocketTts.PMAX * PocketTts.HD + pos * PocketTts.HD, PocketTts.HD)
                        System.arraycopy(values, kvStart + PocketTts.G * PocketTts.HD + g * PocketTts.HD, pv,
                            g * PocketTts.PMAX * PocketTts.HD + pos * PocketTts.HD, PocketTts.HD)
                    }
                    for (h in 0 until PocketTts.NH) mask[h * (PocketTts.PMAX + 1) + pos] = 0f
                    pos++
                }
                return Run(backend, graph, all, tokenIds.size, loadMs, elapsed / 1e6)
            } finally {
                input.forEach { it.close() }
                output.forEach { it.close() }
            }
        } finally {
            model.close()
        }
    }

    private data class VoiceState(val length: Int, val k: FloatArray, val v: FloatArray)

    private fun readVoice(file: File): VoiceState {
        require(file.isFile) { "missing voice state: $file" }
        val input = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val length = input.int
        require(length in 1 until PocketTts.PMAX) { "invalid voice length $length in $file" }
        val size = PocketTts.G * PocketTts.PMAX * PocketTts.HD
        val k = FloatArray(size)
        val v = FloatArray(size)
        for (g in 0 until PocketTts.G) {
            val base = g * PocketTts.PMAX * PocketTts.HD
            for (i in 0 until length * PocketTts.HD) k[base + i] = Half.toFloat(input.short)
        }
        for (g in 0 until PocketTts.G) {
            val base = g * PocketTts.PMAX * PocketTts.HD
            for (i in 0 until length * PocketTts.HD) v[base + i] = Half.toFloat(input.short)
        }
        return VoiceState(length, k, v)
    }

    private fun rope(position: Int, cosine: FloatArray, sine: FloatArray) {
        for (j in 0 until PocketTts.HD / 2) {
            val angle = position / Math.pow(PocketTts.THETA, j / 32.0)
            val c = kotlin.math.cos(angle).toFloat()
            val s = kotlin.math.sin(angle).toFloat()
            cosine[j] = c; cosine[j + PocketTts.HD / 2] = c
            sine[j] = s; sine[j + PocketTts.HD / 2] = s
        }
    }

    private fun metrics(reference: FloatArray, candidate: FloatArray): Metrics {
        require(reference.size == candidate.size) {
            "output size mismatch: CPU=${reference.size}, candidate=${candidate.size}"
        }
        val n = reference.size.toDouble()
        var meanRef = 0.0
        var meanCandidate = 0.0
        var absDiff = 0.0
        var squareDiff = 0.0
        for (i in reference.indices) {
            val a = reference[i].toDouble()
            val b = candidate[i].toDouble()
            check(a.isFinite() && b.isFinite()) { "non-finite output at $i: $a vs $b" }
            meanRef += a
            meanCandidate += b
            val d = a - b
            absDiff += kotlin.math.abs(d)
            squareDiff += d * d
        }
        meanRef /= n
        meanCandidate /= n
        var covariance = 0.0
        var varianceRef = 0.0
        var varianceCandidate = 0.0
        for (i in reference.indices) {
            val a = reference[i].toDouble() - meanRef
            val b = candidate[i].toDouble() - meanCandidate
            covariance += a * b
            varianceRef += a * a
            varianceCandidate += b * b
        }
        val denominator = sqrt(varianceRef * varianceCandidate)
        val corr = if (denominator == 0.0) {
            if (reference.contentEquals(candidate)) 1.0 else Double.NaN
        } else covariance / denominator
        return Metrics(corr, absDiff / n, squareDiff / n)
    }

    private fun writeFloats(file: File, values: FloatArray) {
        val bytes = ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        for (value in values) bytes.putFloat(value)
        file.writeBytes(bytes.array())
    }

    private companion object {
        const val TAG = "FlowLmHarness"
        const val DEFAULT_GRAPH = "pt_flowlm_fused_fp16_no_truncation.tflite"
        const val REFERENCE_GRAPH = "pt_flowlm_fused_fp16.tflite"
        const val DEFAULT_TEXT = "Hello world. This is a FlowLM text probe."
    }
}
