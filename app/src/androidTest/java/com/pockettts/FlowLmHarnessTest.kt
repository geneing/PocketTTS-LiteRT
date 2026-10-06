package com.pockettts

import android.util.Half
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
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

    private data class ModelBuffers(
        val signatureIndex: Int?,
        val inputs: List<TensorBuffer>,
        val outputs: List<TensorBuffer>,
    ) : AutoCloseable {
        override fun close() {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
        }
    }

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

    /**
     * Compares bucketed head-less prompt prefill with sequential steps on the
     * same CPU dynamic-int8 graph. Run this method by itself on the device;
     * it reports every K/V position plus the first post-prompt BOS step.
     */
    @Test
    fun compareBucketedInt8PrefillAgainstSequentialOracle() {
        val graph = args.getString("lmGraph")?.trim()?.ifEmpty { PocketTts.LM } ?: PocketTts.LM
        require(graph == PocketTts.LM) {
            "this acceptance probe is for the CPU dynamic-int8 graph ${PocketTts.LM}, got $graph"
        }
        val voiceNames = args.getString("voices")?.split(',', ';')?.map { it.trim() }?.filter { it.isNotEmpty() }
            ?: listOf("alba", "marius")
        require(voiceNames.size >= 2) { "prefill acceptance requires at least two voices" }
        val sourceText = args.getString("text")?.trim()?.ifEmpty { DEFAULT_PREFILL_TEXT }
            ?: DEFAULT_PREFILL_TEXT
        val baseIds = SpTokenizer(File(modelDir, PocketTts.TOKENIZER)).encode(sourceText)
        require(baseIds.isNotEmpty()) { "text encoded to no tokens" }
        val requestedSizes = args.getString("prefillSizes")?.split(',', ';')?.map { it.trim() }
            ?.filter { it.isNotEmpty() }
            ?: listOf("1", "7", "8", "9", "15", "16", "17", "31", "32", "33", "50", "near")
        val embedBytes = File(modelDir, PocketTts.EMBED).readBytes()
        val embedBuffer = ByteBuffer.wrap(embedBytes).order(ByteOrder.LITTLE_ENDIAN)
        val embedCount = embedBytes.size / 2 / PocketTts.H
        val bos = readFloats(File(modelDir, PocketTts.BOS))
        require(bos.size == PocketTts.H) { "BOS input width ${bos.size}, expected ${PocketTts.H}" }
        val graphFile = File(modelDir, graph)
        check(graphFile.isFile) { "missing graph: $graphFile" }

        val loadStart = System.nanoTime()
        val model = CompiledModel.create(graphFile.absolutePath, CompiledModel.Options(Accelerator.CPU), null)
        val loadMs = (System.nanoTime() - loadStart) / 1e6
        val stepBuffers = createStepBuffers(model)
        check(stepBuffers.signatureIndex == PocketTts.PREFILL_BUCKETS.size) {
            "graph has step signature ${stepBuffers.signatureIndex}; expected " +
                "${PocketTts.PREFILL_BUCKETS.size} after prefill buckets ${PocketTts.PREFILL_BUCKETS.joinToString()}"
        }

        val runDir = File(
            context.getExternalFilesDir("flowlm-harness") ?: File(context.filesDir, "flowlm-harness"),
            "prefill-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())}",
        ).apply { mkdirs() }
        val lines = arrayListOf(
            "CPU int8 bucketed prefill acceptance probe",
            "device=${android.os.Build.MODEL} (${android.os.Build.DEVICE}) Android ${android.os.Build.VERSION.RELEASE}",
            "graph=$graph load_ms=${String.format(Locale.US, "%.2f", loadMs)} step_signature=${stepBuffers.signatureIndex}",
            "prefill_signature_order=${PocketTts.PREFILL_BUCKETS.joinToString()}",
            "buckets choose largest <= remaining tokens; no padding rows",
            "voices=${voiceNames.joinToString()} text=$sourceText",
        )
        val zeroNoise = FloatArray(PocketTts.LDIM)
        try {
            for (voiceName in voiceNames) {
                val voice = readVoice(File(modelDir, PocketTts.voiceFile(voiceName)))
                for (size in requestedSizes) {
                    val count = if (size.equals("near", ignoreCase = true)) {
                        PocketTts.PMAX - voice.length - 1
                    } else size.toInt()
                    require(count > 0 && voice.length + count < PocketTts.PMAX) {
                        "invalid prompt size $size with voice $voiceName length ${voice.length}"
                    }
                    val ids = IntArray(count) { baseIds[it % baseIds.size] }
                    require(ids.all { it in 0 until embedCount }) { "tokenizer id outside embedding table" }
                    val referenceK = Array(count) { FloatArray(PocketTts.G * PocketTts.HD) }
                    val referenceV = Array(count) { FloatArray(PocketTts.G * PocketTts.HD) }
                    var referencePk = voice.k.copyOf()
                    var referencePv = voice.v.copyOf()
                    var position = voice.length
                    var referenceRunNs = 0L
                    val referenceStart = System.nanoTime()
                    for (i in 0 until count) {
                        val emb = embedding(ids[i], embedBuffer)
                        val cosine = FloatArray(PocketTts.HD)
                        val sine = FloatArray(PocketTts.HD)
                        rope(position, cosine, sine)
                        val out = runStep(
                            model, stepBuffers, emb, cosine, sine, stepMask(position),
                            referencePk, referencePv, zeroNoise,
                        ) { elapsed -> referenceRunNs += elapsed }
                        val kBase = 1 + PocketTts.LDIM
                        val vBase = kBase + PocketTts.G * PocketTts.HD
                        System.arraycopy(out, kBase, referenceK[i], 0, referenceK[i].size)
                        System.arraycopy(out, vBase, referenceV[i], 0, referenceV[i].size)
                        putKvRow(referencePk, position, referenceK[i])
                        putKvRow(referencePv, position, referenceV[i])
                        position++
                    }
                    val referenceTotalNs = System.nanoTime() - referenceStart

                    var candidatePk = voice.k.copyOf()
                    var candidatePv = voice.v.copyOf()
                    position = voice.length
                    var i = 0
                    var prefillRunNs = 0L
                    var prefillCalls = 0
                    val bucketPlan = ArrayList<Int>()
                    val prefillStart = System.nanoTime()
                    while (i < count) {
                        val remaining = count - i
                        val bucket = PocketTts.PREFILL_BUCKETS
                            .filter { it <= remaining }.maxOrNull()
                            ?: error("no prefill bucket can cover the remaining $remaining tokens")
                        val signature = PocketTts.prefillSignatureIndex(bucket)
                        check(signature >= 0)
                        val inputs = model.createInputBuffers(signature)
                        val outputs = try {
                            model.createOutputBuffers(signature)
                        } catch (t: Throwable) {
                            inputs.forEach { it.close() }
                            throw t
                        }
                        try {
                            val emb = FloatArray(bucket * PocketTts.H)
                            val cos = FloatArray(bucket * PocketTts.HD)
                            val sin = FloatArray(bucket * PocketTts.HD)
                            val masks = FloatArray(bucket * (PocketTts.PMAX + 1)) { PocketTts.MASK_NEG }
                            val writes = FloatArray(bucket * PocketTts.PMAX)
                            for (j in 0 until bucket) {
                                val rowEmb = embedding(ids[i + j], embedBuffer)
                                System.arraycopy(rowEmb, 0, emb, j * PocketTts.H, PocketTts.H)
                                val c = FloatArray(PocketTts.HD)
                                val s = FloatArray(PocketTts.HD)
                                rope(position + j, c, s)
                                System.arraycopy(c, 0, cos, j * PocketTts.HD, PocketTts.HD)
                                System.arraycopy(s, 0, sin, j * PocketTts.HD, PocketTts.HD)
                                val mb = j * (PocketTts.PMAX + 1)
                                for (q in 0 until position + j) masks[mb + q] = 0f
                                masks[mb + PocketTts.PMAX] = 0f
                                writes[j * PocketTts.PMAX + position + j] = 1f
                            }
                            inputs[0].writeFloat(emb)
                            inputs[1].writeFloat(cos)
                            inputs[2].writeFloat(sin)
                            inputs[3].writeFloat(masks)
                            inputs[4].writeFloat(writes)
                            inputs[5].writeFloat(candidatePk)
                            inputs[6].writeFloat(candidatePv)
                            val runStart = System.nanoTime()
                            model.run(inputs, outputs, signature)
                            prefillRunNs += System.nanoTime() - runStart
                            val out = outputs.single().readFloat()
                            check(out.size == 2 * bucket * PocketTts.G * PocketTts.HD) {
                                "P=$bucket signature $signature returned ${out.size} floats"
                            }
                            val rowSize = PocketTts.G * PocketTts.HD
                            for (j in 0 until bucket) {
                                val kRow = FloatArray(rowSize)
                                val vRow = FloatArray(rowSize)
                                System.arraycopy(out, j * rowSize, kRow, 0, rowSize)
                                System.arraycopy(out, bucket * rowSize + j * rowSize, vRow, 0, rowSize)
                                putKvRow(candidatePk, position + j, kRow)
                                putKvRow(candidatePv, position + j, vRow)
                            }
                            position += bucket
                            i += bucket
                            prefillCalls++
                            bucketPlan += bucket
                        } finally {
                            inputs.forEach { it.close() }
                            outputs.forEach { it.close() }
                        }
                    }
                    val prefillTotalNs = System.nanoTime() - prefillStart
                    val case = "voice=$voiceName tokens=$count"
                    val prefillMsPerToken = prefillTotalNs / 1e6 / count
                    val summaryLine = String.format(
                        Locale.US,
                        "%s buckets=%s reference_total_ms=%.3f reference_run_ms=%.3f " +
                            "prefill_total_ms=%.3f prefill_ms_per_token=%.4f " +
                            "prefill_run_ms=%.3f prefill_calls=%d speedup=%.3fx",
                        case, bucketPlan.joinToString("+"), referenceTotalNs / 1e6, referenceRunNs / 1e6,
                        prefillTotalNs / 1e6, prefillMsPerToken, prefillRunNs / 1e6, prefillCalls,
                        referenceTotalNs.toDouble() / prefillTotalNs,
                    )
                    lines += summaryLine
                    Log.i(TAG, summaryLine)
                    for (row in 0 until count) {
                        val positionLabel = voice.length + row
                        val refK = referenceK[row]
                        val refV = referenceV[row]
                        val candK = getKvRow(candidatePk, positionLabel)
                        val candV = getKvRow(candidatePv, positionLabel)
                        val kDiff = maxDiff(refK, candK)
                        val vDiff = maxDiff(refV, candV)
                        lines += String.format(
                            Locale.US,
                            "  position=%d K corr=%.8f max_abs_diff=%.8g V corr=%.8f max_abs_diff=%.8g",
                            positionLabel, metrics(refK, candK).corr, kDiff,
                            metrics(refV, candV).corr, vDiff,
                        )
                        check(kDiff <= PREFILL_PARITY_TOLERANCE) {
                            "first K divergence: voice=$voiceName tokens=$count position=$positionLabel " +
                                "bucket_plan=${bucketPlan.joinToString("+")} max_abs_diff=$kDiff " +
                                "tolerance=$PREFILL_PARITY_TOLERANCE"
                        }
                        check(vDiff <= PREFILL_PARITY_TOLERANCE) {
                            "first V divergence: voice=$voiceName tokens=$count position=$positionLabel " +
                                "bucket_plan=${bucketPlan.joinToString("+")} max_abs_diff=$vDiff " +
                                "tolerance=$PREFILL_PARITY_TOLERANCE"
                        }
                    }
                    val prefixKDiff = maxPrefixDiff(candidatePk, voice.k, voice.length)
                    val prefixVDiff = maxPrefixDiff(candidatePv, voice.v, voice.length)
                    lines += "  voice_prefix K max_abs_diff=$prefixKDiff V max_abs_diff=$prefixVDiff"
                    check(prefixKDiff == 0.0 && prefixVDiff == 0.0) {
                        "prefill modified voice prefix for $voiceName/$count"
                    }

                    val firstPos = voice.length + count
                    val firstCos = FloatArray(PocketTts.HD)
                    val firstSin = FloatArray(PocketTts.HD)
                    rope(firstPos, firstCos, firstSin)
                    val firstRef = runStep(
                        model, stepBuffers, bos, firstCos, firstSin, stepMask(firstPos),
                        referencePk, referencePv, zeroNoise,
                    )
                    val firstCandidate = runStep(
                        model, stepBuffers, bos, firstCos, firstSin, stepMask(firstPos),
                        candidatePk, candidatePv, zeroNoise,
                    )
                    val firstDecodeLine = String.format(
                        Locale.US,
                        "  first_decode eos_diff=%.8g latent_corr=%.8f latent_max_abs_diff=%.8g " +
                            "new_k_max_abs_diff=%.8g new_v_max_abs_diff=%.8g",
                        maxDiff(firstRef.copyOfRange(0, 1), firstCandidate.copyOfRange(0, 1)),
                        metrics(firstRef.copyOfRange(1, 1 + PocketTts.LDIM),
                            firstCandidate.copyOfRange(1, 1 + PocketTts.LDIM)).corr,
                        maxDiff(firstRef.copyOfRange(1, 1 + PocketTts.LDIM),
                            firstCandidate.copyOfRange(1, 1 + PocketTts.LDIM)),
                        maxDiff(firstRef.copyOfRange(1 + PocketTts.LDIM,
                            1 + PocketTts.LDIM + PocketTts.G * PocketTts.HD),
                            firstCandidate.copyOfRange(1 + PocketTts.LDIM,
                                1 + PocketTts.LDIM + PocketTts.G * PocketTts.HD)),
                        maxDiff(firstRef.copyOfRange(1 + PocketTts.LDIM + PocketTts.G * PocketTts.HD,
                            firstRef.size),
                            firstCandidate.copyOfRange(1 + PocketTts.LDIM + PocketTts.G * PocketTts.HD,
                                firstCandidate.size)),
                    )
                    lines += firstDecodeLine
                    Log.i(TAG, firstDecodeLine)
                    val eosDiff = maxDiff(firstRef.copyOfRange(0, 1), firstCandidate.copyOfRange(0, 1))
                    val latentDiff = maxDiff(
                        firstRef.copyOfRange(1, 1 + PocketTts.LDIM),
                        firstCandidate.copyOfRange(1, 1 + PocketTts.LDIM),
                    )
                    val firstKDiff = maxDiff(
                        firstRef.copyOfRange(1 + PocketTts.LDIM,
                            1 + PocketTts.LDIM + PocketTts.G * PocketTts.HD),
                        firstCandidate.copyOfRange(1 + PocketTts.LDIM,
                            1 + PocketTts.LDIM + PocketTts.G * PocketTts.HD),
                    )
                    val firstVDiff = maxDiff(
                        firstRef.copyOfRange(1 + PocketTts.LDIM + PocketTts.G * PocketTts.HD,
                            firstRef.size),
                        firstCandidate.copyOfRange(1 + PocketTts.LDIM + PocketTts.G * PocketTts.HD,
                            firstCandidate.size),
                    )
                    check(eosDiff <= PREFILL_PARITY_TOLERANCE &&
                        latentDiff <= PREFILL_PARITY_TOLERANCE &&
                        firstKDiff <= PREFILL_PARITY_TOLERANCE &&
                        firstVDiff <= PREFILL_PARITY_TOLERANCE) {
                        "first post-prefill BOS decode diverged for voice=$voiceName tokens=$count: " +
                            "eos=$eosDiff latent=$latentDiff K=$firstKDiff V=$firstVDiff " +
                            "tolerance=$PREFILL_PARITY_TOLERANCE"
                    }
                }
            }
            val report = lines.joinToString("\n", postfix = "\n")
            File(runDir, "report.txt").writeText(report)
            Log.i(TAG, "prefill detail report saved to ${File(runDir, "report.txt").absolutePath}")
            assertTrue("CPU int8 prefill produced no measurements", lines.size > 6)
        } finally {
            stepBuffers.close()
            model.close()
        }
    }

    private fun createStepBuffers(model: CompiledModel): ModelBuffers {
        for (index in listOf(PocketTts.PREFILL_BUCKETS.size, 1).distinct()) {
            val inputs = runCatching { model.createInputBuffers(index) }.getOrNull() ?: continue
            val outputs = try {
                model.createOutputBuffers(index)
            } catch (t: Throwable) {
                inputs.forEach { it.close() }
                throw t
            }
            return ModelBuffers(index, inputs, outputs)
        }
        return ModelBuffers(null, model.createInputBuffers(), model.createOutputBuffers())
    }

    private fun run(model: CompiledModel, buffers: ModelBuffers) {
        val index = buffers.signatureIndex
        if (index == null) model.run(buffers.inputs, buffers.outputs)
        else model.run(buffers.inputs, buffers.outputs, index)
    }

    private fun runStep(
        model: CompiledModel,
        buffers: ModelBuffers,
        emb: FloatArray,
        cosine: FloatArray,
        sine: FloatArray,
        mask: FloatArray,
        pk: FloatArray,
        pv: FloatArray,
        noise: FloatArray,
        onRun: (Long) -> Unit = {},
    ): FloatArray {
        val input = buffers.inputs
        input[0].writeFloat(emb)
        input[1].writeFloat(cosine)
        input[2].writeFloat(sine)
        input[3].writeFloat(mask)
        input[4].writeFloat(pk)
        input[5].writeFloat(pv)
        input[6].writeFloat(noise)
        val runStart = System.nanoTime()
        run(model, buffers)
        onRun(System.nanoTime() - runStart)
        return buffers.outputs.single().readFloat()
    }

    private fun embedding(id: Int, source: ByteBuffer): FloatArray {
        require(id >= 0)
        val out = FloatArray(PocketTts.H)
        var offset = id * PocketTts.H * Short.SIZE_BYTES
        for (i in out.indices) {
            out[i] = Half.toFloat(source.getShort(offset))
            offset += Short.SIZE_BYTES
        }
        return out
    }

    private fun stepMask(position: Int): FloatArray {
        val mask = FloatArray(PocketTts.NH * (PocketTts.PMAX + 1)) { PocketTts.MASK_NEG }
        for (h in 0 until PocketTts.NH) {
            val row = h * (PocketTts.PMAX + 1)
            for (q in 0 until position) mask[row + q] = 0f
            mask[row + PocketTts.PMAX] = 0f
        }
        return mask
    }

    private fun putKvRow(cache: FloatArray, position: Int, row: FloatArray) {
        require(row.size == PocketTts.G * PocketTts.HD)
        for (g in 0 until PocketTts.G) {
            System.arraycopy(row, g * PocketTts.HD, cache,
                g * PocketTts.PMAX * PocketTts.HD + position * PocketTts.HD, PocketTts.HD)
        }
    }

    private fun getKvRow(cache: FloatArray, position: Int): FloatArray {
        val row = FloatArray(PocketTts.G * PocketTts.HD)
        for (g in 0 until PocketTts.G) {
            System.arraycopy(cache, g * PocketTts.PMAX * PocketTts.HD + position * PocketTts.HD,
                row, g * PocketTts.HD, PocketTts.HD)
        }
        return row
    }

    private fun maxPrefixDiff(a: FloatArray, b: FloatArray, positions: Int): Double {
        var max = 0.0
        for (g in 0 until PocketTts.G) {
            val start = g * PocketTts.PMAX * PocketTts.HD
            for (i in 0 until positions * PocketTts.HD) {
                max = maxOf(max, kotlin.math.abs(a[start + i].toDouble() - b[start + i].toDouble()))
            }
        }
        return max
    }

    private fun readFloats(file: File): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size % Float.SIZE_BYTES == 0) { "invalid float file: $file" }
        val input = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / Float.SIZE_BYTES) { input.float }
    }

    private fun maxDiff(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size) { "diff size mismatch: ${a.size} vs ${b.size}" }
        var max = 0.0
        for (i in a.indices) {
            val x = a[i].toDouble()
            val y = b[i].toDouble()
            check(x.isFinite() && y.isFinite()) { "non-finite value at $i: $x vs $y" }
            max = maxOf(max, kotlin.math.abs(x - y))
        }
        return max
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
            val stepBuffers = createStepBuffers(model)
            val input = stepBuffers.inputs
            val output = stepBuffers.outputs
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
                    run(model, stepBuffers)
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
                stepBuffers.close()
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
        const val PREFILL_PARITY_TOLERANCE = 1e-5
        const val DEFAULT_GRAPH = "pt_flowlm_fused_fp16_no_truncation.tflite"
        const val REFERENCE_GRAPH = "pt_flowlm_fused_fp16.tflite"
        const val DEFAULT_TEXT = "Hello world. This is a FlowLM text probe."
        const val DEFAULT_PREFILL_TEXT = """
            Pocket TTS reads this repeated paragraph to exercise prompt prefill buckets on a phone.
            It keeps each voice prefix intact, follows the absolute rotary positions, and checks that
            every causal key and value row matches the token by token CPU int8 reference.
            Then it runs the first audio step from the same beginning of sequence input and reports
            timing, cache differences, latent differences, and the selected fixed bucket sizes.
        """
    }
}
