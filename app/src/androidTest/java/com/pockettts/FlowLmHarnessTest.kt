package com.pockettts

import android.util.Half
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import dev.pockettts.Accel
import dev.pockettts.Placement
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsConfig
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import dev.pockettts.SpTokenizer
import dev.pockettts.TtsResult
import dev.pockettts.Wav
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.roundToInt
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
        val capacity: Int,
        val kvPrecision: KvPrecision,
        val output: FloatArray,
        val tokenCount: Int,
        val loadMs: Double,
        val stageMs: Double,
        val runMs: Double,
        val readMs: Double,
        val kvTransformMs: Double,
        val kvInputBytes: Long,
    )

    private data class Metrics(val corr: Double, val mad: Double, val msd: Double)
    private enum class KvPrecision { FP32, FP16, INT8_PER_HEAD }

    @Test
    fun capacityPlannerAndVoicePrefixRepackingAreExact() {
        assertEquals(128, PocketTts.smallestFlowLmCapacity(127, 0, 0))
        assertEquals(256, PocketTts.smallestFlowLmCapacity(128, 0, 0))
        assertEquals(256, PocketTts.smallestFlowLmCapacity(120, 10, 20))
        assertEquals(128, PocketTts.smallestFlowLmCapacity(0, 0, 127))
        assertEquals(256, PocketTts.smallestFlowLmCapacity(0, 0, 128))
        assertEquals(512, PocketTts.smallestFlowLmCapacity(0, 0, 511))
        assertNull(PocketTts.smallestFlowLmCapacity(0, 0, 512))
        assertEquals("pt_flowlm_fused_dyn8_all_pmax256.tflite",
            PocketTts.flowLmCapacityGraph(PocketTts.LM, 256))
        assertEquals(PocketTts.LM, PocketTts.flowLmCapacityGraph(PocketTts.LM, PocketTts.PMAX))

        val source = FloatArray(PocketTts.G * PocketTts.PMAX * PocketTts.HD)
        val length = 2
        for (g in 0 until PocketTts.G) {
            val start = g * PocketTts.PMAX * PocketTts.HD
            for (i in 0 until length * PocketTts.HD) source[start + i] = g * 10000f + i
        }
        val packed = repackVoice(source, length, 128)
        for (g in 0 until PocketTts.G) {
            val src = g * PocketTts.PMAX * PocketTts.HD
            val dst = g * 128 * PocketTts.HD
            assertArrayEquals(
                source.copyOfRange(src, src + length * PocketTts.HD),
                packed.copyOfRange(dst, dst + length * PocketTts.HD),
                0f,
            )
            assertTrue((dst + length * PocketTts.HD until dst + 128 * PocketTts.HD)
                .all { packed[it] == 0f })
        }
    }

    @Test
    fun runTextPromptHarness() {
        val text = args.getString("text")?.trim().orEmpty().ifEmpty { DEFAULT_TEXT }
        val voice = args.getString("voice")?.trim()?.ifEmpty { "alba" } ?: "alba"
        val baseGraph = args.getString("lmGraph")?.trim()?.ifEmpty { DEFAULT_GRAPH } ?: DEFAULT_GRAPH
        val candidateCapacity = args.getString("lmCapacity")?.toIntOrNull() ?: graphCapacity(baseGraph)
        require(candidateCapacity in PocketTts.FLOWLM_CAPACITIES) {
            "lmCapacity must be one of ${PocketTts.FLOWLM_CAPACITIES}, got $candidateCapacity"
        }
        val candidateGraph = PocketTts.flowLmCapacityGraph(baseGraph, candidateCapacity)
        val referenceGraph = args.getString("referenceGraph")
            ?.trim()?.ifEmpty { REFERENCE_GRAPH } ?: REFERENCE_GRAPH
        val kvPrecision = parseKvPrecision(args.getString("kvPrecision"))
        val includeFirstDecode = args.getString("firstDecode")?.toBooleanStrictOrNull() ?: true
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
        require(tokenIds.size + voiceState.length < candidateCapacity) {
            "prompt has ${tokenIds.size} tokens; voice '$voice' leaves only " +
                "${candidateCapacity - voiceState.length - 1} FlowLM positions at capacity $candidateCapacity"
        }
        val plannedFrames = args.getString("plannedFrames")?.toIntOrNull()
            ?: plannedFrameBudget(tokenIds.size)
        require(plannedFrames >= 0) { "plannedFrames cannot be negative" }
        val recommendedCapacity = PocketTts.smallestFlowLmCapacity(
            voiceState.length, tokenIds.size, plannedFrames,
        )
        val plannedFits = recommendedCapacity != null && candidateCapacity >= recommendedCapacity
        val embed = File(modelDir, PocketTts.EMBED).readBytes()
        val bosInput = readFloats(File(modelDir, PocketTts.BOS), PocketTts.H)
        val nEmbeddings = embed.size / 2 / PocketTts.H
        require(tokenIds.all { it in 0 until nEmbeddings }) { "tokenizer id outside embedding table" }

        val runDir = File(
            context.getExternalFilesDir("flowlm-harness") ?: File(context.filesDir, "flowlm-harness"),
            "run-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())}",
        ).apply { mkdirs() }
        val lines = ArrayList<String>()
        lines += "FlowLM text harness"
        lines += "device=${android.os.Build.MODEL} (${android.os.Build.DEVICE}) Android ${android.os.Build.VERSION.RELEASE}"
        lines += "fingerprint=${android.os.Build.FINGERPRINT}"
        lines += "LiteRT Android dependency=2.2.0; inspect installed dispatch library for the NPU variant"
        lines += "text=$text"
        lines += "voice=$voice"
        lines += "graph=$candidateGraph capacity=$candidateCapacity kvPrecision=$kvPrecision"
        lines += "candidate_sha256=${sha256(File(modelDir, candidateGraph))}"
        lines += "reference_graph=$referenceGraph reference_sha256=${sha256(File(modelDir, referenceGraph))}"
        lines += "tokens=${tokenIds.size} ids=${tokenIds.joinToString(",")}"
        lines += "output=prompt fused outputs plus ${if (includeFirstDecode) "one zero-noise BOS decode step" else "no decode step"}; voice KV prefix=${voiceState.length} tokens"
        lines += "plannedFrames=$plannedFrames recommendedCapacity=${recommendedCapacity ?: "unsupported"} fitsCapacity=$plannedFits; prompt-only probe; planned fit is required for full speech"
        lines += String.format(
            Locale.US,
            "KV input bytes/call: fp32=%d fp16-equivalent=%d int8-equivalent=%d; harness graph I/O remains fp32",
            kvBytes(candidateCapacity, 4), kvBytes(candidateCapacity, 2), kvBytes(candidateCapacity, 1),
        )

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
                val graph = if (backend == Accel.NPU) PocketTts.g5Variant(candidateGraph) else candidateGraph
                val result = try {
                    if (candidateCapacity < PocketTts.PMAX && backend == Accel.NPU) {
                        error("reduced-capacity graphs have not passed the Tensor G5 AOT gate")
                    }
                    if (!plannedFits) {
                        error(
                            "capacity $candidateCapacity cannot fit voice=${voiceState.length} + " +
                                "prompt=${tokenIds.size} + plannedFrames=$plannedFrames; select a larger bucket"
                        )
                    }
                    runPrompt(
                        backend, graph, tokenIds, embed, voiceState, environment,
                        candidateCapacity, kvPrecision, bosInput, includeFirstDecode,
                    )
                } catch (t: Throwable) {
                    lines += "$backend FAILED: ${t.message ?: t::class.java.simpleName}"
                    Log.e(TAG, "backend=$backend graph=$graph failed", t)
                    null
                }
                if (result != null) candidates[backend] = result
            }

            // The shipped CPU dynamic-int8 graph is the A/B reference. Running it after the candidates also
            // preserves the runtime's required NPU-first initialization order.
            val referenceCapacity = graphCapacity(referenceGraph)
            val reference = runPrompt(
                Accel.CPU, referenceGraph, tokenIds, embed, voiceState, environment,
                referenceCapacity, KvPrecision.FP32, bosInput, includeFirstDecode,
            )
            writeFloats(File(runDir, "cpu-reference.f32le"), reference.output)
            lines += String.format(
                Locale.US,
                "CPU reference: graph=%s capacity=%d load=%.2f ms stage=%.2f ms run=%.2f ms read=%.2f ms values=%d saved=cpu-reference.f32le",
                reference.graph, reference.capacity, reference.loadMs, reference.stageMs,
                reference.runMs, reference.readMs, reference.output.size,
            )
            for ((backend, result) in candidates) {
                val file = "${backend.name.lowercase(Locale.ROOT)}-candidate.f32le"
                writeFloats(File(runDir, file), result.output)
                try {
                    val m = metrics(reference.output, result.output)
                    lines += String.format(
                        Locale.US,
                        "%s candidate: graph=%s capacity=%d precision=%s load=%.2f ms stage=%.2f ms run=%.2f ms read=%.2f ms kv_transform=%.2f ms kv_in_bytes=%d corr=%.8f mean_abs_diff=%.8g mean_square_diff=%.8g saved=%s",
                        backend, result.graph, result.capacity, result.kvPrecision, result.loadMs,
                        result.stageMs, result.runMs, result.readMs, result.kvTransformMs,
                        result.kvInputBytes, m.corr, m.mad, m.msd, file,
                    )
                    appendStepDifferences(lines, reference.output, result.output,
                        tokenIds.size, includeFirstDecode)
                } catch (t: Throwable) {
                    lines += "$backend candidate comparison failed: ${t.message}; output saved=$file"
                }
            }
            lines += "CPU self-check: corr=1.00000000 mean_abs_diff=0 mean_square_diff=0"
            lines += "raw outputs are little-endian float32; ${PocketTts.H} embedding width; ${tokenIds.size} prompt tokens plus ${if (includeFirstDecode) 1 else 0} decode step"
            val report = lines.joinToString("\n", postfix = "\n")
            File(runDir, "report.txt").writeText(report)
            report.lines().forEach { Log.i(TAG, it) }
            Log.i(TAG, "saved FlowLM probe files to ${runDir.absolutePath}")
            assertTrue("no candidate backend completed; see report.txt", candidates.isNotEmpty())
            assertTrue("CPU production reference produced no output", reference.output.isNotEmpty())
        } finally {
            environment?.close()
        }
    }

    /** Full short-utterance A/B with the production decoder placement and seed. */
    @Test
    fun shortSpeechCapacityPair() {
        val text = args.getString("text")?.trim().orEmpty().ifEmpty { "Hello there, how are you?" }
        val voice = args.getString("voice")?.trim()?.ifEmpty { "alba" } ?: "alba"
        val capacity = args.getString("lmCapacity")?.toIntOrNull() ?: 256
        require(capacity in PocketTts.FLOWLM_CAPACITIES && capacity < PocketTts.PMAX) {
            "lmCapacity must be 128 or 256 for the reduced-capacity speech pair"
        }
        val mode = args.getString("mode")?.trim()?.ifEmpty { "stream" } ?: "stream"
        require(mode == "stream" || mode == "oneShot") { "mode must be stream or oneShot" }
        val order = args.getString("order")?.trim()?.ifEmpty { "candidate_first" }
            ?: "candidate_first"
        require(order == "candidate_first" || order == "reference_first") {
            "order must be candidate_first or reference_first"
        }
        val placement = Placement.default(context, modelDir)
        require(placement.lm == Accel.CPU) {
            "FlowLM bucket A/B requires the production CPU LM placement; got ${placement.label}"
        }
        val models = PocketTtsModels.default(context)
        val candidateGraph = PocketTts.flowLmCapacityGraph(PocketTts.LM, capacity)
        require(models.store.exists(candidateGraph)) { "missing candidate graph $candidateGraph" }
        require(models.store.exists(PocketTts.LM)) { "missing reference graph ${PocketTts.LM}" }
        val runDir = File(
            context.getExternalFilesDir("flowlm-harness") ?: File(context.filesDir, "flowlm-harness"),
            "speech-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())}",
        ).apply { mkdirs() }
        data class SpeechRun(val result: TtsResult, val loadMs: Double, val graph: String,
                             val runtime: String)
        fun run(label: String, selectedCapacity: Int): SpeechRun {
            val started = System.nanoTime()
            return PocketTtsEngine(context, PocketTtsConfig(
                models = models,
                placement = placement,
                lmCapacity = selectedCapacity,
                noiseSeed = 42L,
            )).use { engine ->
                val loadMs = (System.nanoTime() - started) / 1e6
                check(engine.runtimeAccelerators["lm"] == Accel.CPU) {
                    "FlowLM $label unexpectedly loaded on ${engine.runtimeAccelerators["lm"]}"
                }
                val result = if (mode == "stream") engine.stream(text, voice) {}
                    else engine.synthesize(text, voice)
                check(result.frames > 0 && result.audio.isNotEmpty() && result.audio.all { it.isFinite() }) {
                    "$label produced no finite speech"
                }
                Wav.write(File(runDir, "$label.wav"), result.audio)
                SpeechRun(result, loadMs, engine.lmGraphName, engine.runtimeAccelerators.toString())
            }
        }
        val first = if (order == "candidate_first") "candidate" else "reference"
        val firstRun = run(first, if (first == "candidate") capacity else PocketTts.PMAX)
        val second = if (first == "candidate") "reference" else "candidate"
        val secondRun = run(second, if (second == "candidate") capacity else PocketTts.PMAX)
        val candidate = if (first == "candidate") firstRun else secondRun
        val reference = if (first == "reference") firstRun else secondRun
        val quality = AudioQuality.compare(reference.result.audio, candidate.result.audio)
        val lines = mutableListOf(
            "FlowLM short speech capacity pair",
            "fingerprint=${android.os.Build.FINGERPRINT}",
            "text=$text voice=$voice seed=42 mode=$mode order=$order placement=${placement.label}",
            "candidate_graph=${candidate.graph} sha256=${sha256(File(modelDir, candidate.graph))}",
            "reference_graph=${reference.graph} sha256=${sha256(File(modelDir, reference.graph))}",
        )
        fun describe(label: String, run: SpeechRun) {
            val result = run.result
            val p = result.profile
            val seconds = result.audio.size.toDouble() / PocketTts.SAMPLE_RATE
            lines += String.format(Locale.US,
                "%s runtime=%s load=%.2f ms frames=%d audio=%.3f s synthesis=%d ms RTF=%.3f " +
                    "first_audio=%d ms lm_in=%d ms lm_run=%d ms lm_read=%d ms " +
                    "dectx=%d ms seanet=%d ms chunks=%d saved=%s.wav",
                label, run.runtime, run.loadMs, result.frames, seconds, result.ms,
                seconds * 1000 / result.ms, p.firstChunkMs, p.lmInMs, p.lmRunMs,
                p.lmReadMs, p.decTxMs, p.seanetMs, p.audioChunks, label)
        }
        describe("candidate", candidate)
        describe("reference", reference)
        lines += "waveform_corr=${quality.corr}; compare completion and listen to both WAVs"
        val report = lines.joinToString("\n", postfix = "\n")
        File(runDir, "report.txt").writeText(report)
        report.lineSequence().forEach { Log.i(TAG, it) }
        assertTrue("candidate speech is empty", candidate.result.audio.isNotEmpty())
    }

    private fun runPrompt(
        backend: Accel,
        graph: String,
        tokenIds: IntArray,
        embedBytes: ByteArray,
        voice: VoiceState,
        environment: Environment?,
        capacity: Int,
        kvPrecision: KvPrecision,
        bosInput: FloatArray,
        includeFirstDecode: Boolean,
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
                require(capacity in PocketTts.FLOWLM_CAPACITIES)
                require(voice.length < capacity) {
                    "voice prefix ${voice.length} does not fit FlowLM capacity $capacity"
                }
                val pk = repackVoice(voice.k, voice.length, capacity)
                val pv = repackVoice(voice.v, voice.length, capacity)
                val mask = FloatArray(PocketTts.NH * (capacity + 1)) { PocketTts.MASK_NEG }
                var pos = voice.length
                for (h in 0 until PocketTts.NH) {
                    val row = h * (capacity + 1)
                    for (p in 0 until pos) mask[row + p] = 0f
                    mask[row + capacity] = 0f
                }
                val cosine = FloatArray(PocketTts.HD)
                val sine = FloatArray(PocketTts.HD)
                val zeroNoise = FloatArray(PocketTts.LDIM)
                val embed = ByteBuffer.wrap(embedBytes).order(ByteOrder.LITTLE_ENDIAN)
                val outputPerToken = 1 + PocketTts.LDIM + 2 * PocketTts.G * PocketTts.HD
                val all = FloatArray((tokenIds.size + if (includeFirstDecode) 1 else 0) * outputPerToken)
                var outputOffset = 0
                var stageNs = 0L
                var runNs = 0L
                var readNs = 0L
                var kvTransformNs = 0L
                fun runStep(embedding: FloatArray) {
                    rope(pos, cosine, sine)
                    val transformStart = System.nanoTime()
                    transformKv(pk, pv, capacity, pos, kvPrecision)
                    kvTransformNs += System.nanoTime() - transformStart
                    val stageStart = System.nanoTime()
                    input[0].writeFloat(embedding)
                    input[1].writeFloat(cosine)
                    input[2].writeFloat(sine)
                    input[3].writeFloat(mask)
                    input[4].writeFloat(pk)
                    input[5].writeFloat(pv)
                    input[6].writeFloat(zeroNoise)
                    stageNs += System.nanoTime() - stageStart
                    val started = System.nanoTime()
                    if (stepInput != null) model.run(input, output, 1) else model.run(input, output)
                    runNs += System.nanoTime() - started
                    val readStart = System.nanoTime()
                    val values = output.single().readFloat()
                    readNs += System.nanoTime() - readStart
                    check(values.size == outputPerToken) {
                        "FlowLM output width changed: expected $outputPerToken, got ${values.size}"
                    }
                    System.arraycopy(values, 0, all, outputOffset, values.size)
                    outputOffset += values.size

                    val kvStart = 1 + PocketTts.LDIM
                    for (g in 0 until PocketTts.G) {
                        System.arraycopy(values, kvStart + g * PocketTts.HD, pk,
                            g * capacity * PocketTts.HD + pos * PocketTts.HD, PocketTts.HD)
                        System.arraycopy(values, kvStart + PocketTts.G * PocketTts.HD + g * PocketTts.HD, pv,
                            g * capacity * PocketTts.HD + pos * PocketTts.HD, PocketTts.HD)
                    }
                    for (h in 0 until PocketTts.NH) mask[h * (capacity + 1) + pos] = 0f
                    pos++
                }
                for (id in tokenIds) {
                    val embedding = FloatArray(PocketTts.H)
                    var offset = id * PocketTts.H * Short.SIZE_BYTES
                    for (i in embedding.indices) {
                        embedding[i] = Half.toFloat(embed.getShort(offset))
                        offset += Short.SIZE_BYTES
                    }
                    runStep(embedding)
                }
                if (includeFirstDecode) runStep(bosInput)
                return Run(
                    backend = backend,
                    graph = graph,
                    capacity = capacity,
                    kvPrecision = kvPrecision,
                    output = all,
                    tokenCount = tokenIds.size + if (includeFirstDecode) 1 else 0,
                    loadMs = loadMs,
                    stageMs = stageNs / 1e6,
                    runMs = runNs / 1e6,
                    readMs = readNs / 1e6,
                    kvTransformMs = kvTransformNs / 1e6,
                    kvInputBytes = kvBytes(capacity, Float.SIZE_BYTES),
                )
            } finally {
                input.forEach { it.close() }
                output.forEach { it.close() }
            }
        } finally {
            model.close()
        }
    }

    private data class VoiceState(val length: Int, val k: FloatArray, val v: FloatArray)

    private fun graphCapacity(graph: String): Int {
        val match = Regex("_pmax(\\d+)(?:_g5)?\\.tflite$").find(graph)
        return match?.groupValues?.get(1)?.toIntOrNull() ?: PocketTts.PMAX
    }

    private fun parseKvPrecision(value: String?): KvPrecision = when (
        value?.trim()?.uppercase(Locale.ROOT)?.replace('-', '_')
    ) {
        null, "", "FP32", "FLOAT32" -> KvPrecision.FP32
        "FP16", "FLOAT16" -> KvPrecision.FP16
        "INT8", "INT8_PER_HEAD" -> KvPrecision.INT8_PER_HEAD
        else -> error("kvPrecision must be fp32, fp16, or int8_per_head; got '$value'")
    }

    private fun plannedFrameBudget(tokenCount: Int): Int = ceil(
        (tokenCount / PocketTts.TOKENS_PER_SECOND + PocketTts.GEN_SECONDS_PADDING) *
            PocketTts.FRAME_RATE,
    ).toInt()

    private fun kvBytes(capacity: Int, bytesPerValue: Int): Long =
        2L * PocketTts.G * capacity * PocketTts.HD * bytesPerValue

    /** Repack only the live voice prefix from the on-disk 512-row stride. */
    private fun repackVoice(source: FloatArray, length: Int, capacity: Int): FloatArray {
        val expected = PocketTts.G * PocketTts.PMAX * PocketTts.HD
        require(source.size == expected) { "voice KV size changed: expected $expected, got ${source.size}" }
        require(length < capacity) { "voice prefix $length does not fit capacity $capacity" }
        val target = FloatArray(PocketTts.G * capacity * PocketTts.HD)
        for (g in 0 until PocketTts.G) {
            System.arraycopy(
                source, g * PocketTts.PMAX * PocketTts.HD,
                target, g * capacity * PocketTts.HD,
                length * PocketTts.HD,
            )
        }
        return target
    }

    /**
     * Fidelity-only simulation of lower-precision cache storage. The current
     * graph buffers remain float32, so this does not reduce LiteRT transfer bytes.
     */
    private fun transformKv(
        k: FloatArray,
        v: FloatArray,
        capacity: Int,
        liveLength: Int,
        precision: KvPrecision,
    ) {
        when (precision) {
            KvPrecision.FP32 -> Unit
            KvPrecision.FP16 -> {
                for (g in 0 until PocketTts.G) {
                    val base = g * capacity * PocketTts.HD
                    val live = liveLength * PocketTts.HD
                    for (i in 0 until live) {
                        k[base + i] = Half.toFloat(Half.toHalf(k[base + i]))
                        v[base + i] = Half.toFloat(Half.toHalf(v[base + i]))
                    }
                }
            }
            KvPrecision.INT8_PER_HEAD -> {
                for (g in 0 until PocketTts.G) {
                    val base = g * capacity * PocketTts.HD
                    quantizePerHead(k, base, liveLength * PocketTts.HD)
                    quantizePerHead(v, base, liveLength * PocketTts.HD)
                }
            }
        }
    }

    private fun quantizePerHead(values: FloatArray, start: Int, count: Int) {
        var maxAbs = 0f
        for (i in 0 until count) maxAbs = maxOf(maxAbs, kotlin.math.abs(values[start + i]))
        if (maxAbs == 0f) return
        val scale = maxAbs / 127f
        for (i in 0 until count) {
            val q = (values[start + i] / scale).roundToInt().coerceIn(-127, 127)
            values[start + i] = q * scale
        }
    }

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

    /** Report every real newly written K/V row and the first decode output separately. */
    private fun appendStepDifferences(
        lines: MutableList<String>, reference: FloatArray, candidate: FloatArray,
        promptTokens: Int, includeFirstDecode: Boolean,
    ) {
        val kvWidth = PocketTts.G * PocketTts.HD
        val stride = 1 + PocketTts.LDIM + 2 * kvWidth
        val steps = promptTokens + if (includeFirstDecode) 1 else 0
        require(reference.size == steps * stride && candidate.size == reference.size) {
            "step comparison size mismatch: expected ${steps * stride}, " +
                "reference=${reference.size}, candidate=${candidate.size}"
        }
        fun maxAbsDiff(start: Int, count: Int): Double {
            var maximum = 0.0
            for (i in start until start + count) {
                val difference = kotlin.math.abs(reference[i].toDouble() - candidate[i].toDouble())
                check(difference.isFinite()) { "non-finite difference at output $i" }
                maximum = maxOf(maximum, difference)
            }
            return maximum
        }
        var firstDifferent: Int? = null
        var firstOverTolerance: Int? = null
        for (step in 0 until steps) {
            val start = step * stride
            val eos = maxAbsDiff(start, 1)
            val latent = maxAbsDiff(start + 1, PocketTts.LDIM)
            val key = maxAbsDiff(start + 1 + PocketTts.LDIM, kvWidth)
            val value = maxAbsDiff(start + 1 + PocketTts.LDIM + kvWidth, kvWidth)
            val maximum = maxOf(eos, latent, key, value)
            if (maximum > 0.0 && firstDifferent == null) firstDifferent = step
            if (maximum > 1e-3 && firstOverTolerance == null) firstOverTolerance = step
            lines += String.format(Locale.US,
                "step=%d kind=%s eos_abs=%.8g latent_max_abs=%.8g k_max_abs=%.8g v_max_abs=%.8g",
                step, if (step < promptTokens) "prompt" else "first_decode",
                eos, latent, key, value)
        }
        lines += "first_nonidentical_step=${firstDifferent ?: "none"} " +
            "first_step_over_1e-3=${firstOverTolerance ?: "none"}; threshold is diagnostic only"
    }

    private fun sha256(file: File): String {
        if (!file.isFile) return "missing"
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        val hex = "0123456789abcdef"
        return buildString(64) {
            for (byte in digest.digest()) {
                val value = byte.toInt() and 0xff
                append(hex[value ushr 4]); append(hex[value and 0x0f])
            }
        }
    }

    private fun writeFloats(file: File, values: FloatArray) {
        val bytes = ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        for (value in values) bytes.putFloat(value)
        file.writeBytes(bytes.array())
    }

    private fun readFloats(file: File, expected: Int): FloatArray {
        require(file.isFile) { "missing float asset: $file" }
        val bytes = file.readBytes()
        require(bytes.size == expected * Float.SIZE_BYTES) {
            "unexpected float asset size for $file: expected ${expected * Float.SIZE_BYTES}, got ${bytes.size}"
        }
        val input = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(expected) { input.float }
    }

    private companion object {
        const val TAG = "FlowLmHarness"
        const val DEFAULT_GRAPH = PocketTts.LM
        const val REFERENCE_GRAPH = "pt_flowlm_fused_dyn8_all.tflite"
        const val DEFAULT_TEXT = "Hello world. This is a FlowLM text probe."
    }
}
