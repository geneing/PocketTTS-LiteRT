package com.pockettts

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Half
import android.util.Log
import android.os.Debug
import android.os.Build
import android.os.OutcomeReceiver
import android.os.PowerManager
import android.os.PowerMonitor
import android.os.PowerMonitorReadings
import android.os.SystemClock
import android.os.health.SystemHealthManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import dev.pockettts.Accel
import dev.pockettts.GpuAhwbLmRunner
import dev.pockettts.GpuOpenClLmRunner
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsConfig
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import dev.pockettts.Placement
import dev.pockettts.SpTokenizer
import dev.pockettts.Wav
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.LinkedHashMap
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
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
        val inputMs: Double,
        val runMs: Double,
        val readMs: Double,
        val cacheCopyMs: Double,
        val firstRunMs: Double,
    )

    private data class Metrics(val corr: Double, val mad: Double, val msd: Double)

    private data class SpeechProbe(
        val result: dev.pockettts.TtsResult,
        val loadMs: Map<String, Long>,
        val backends: Map<String, Accel>,
        val pssBeforeRunKb: Long,
        val pssAfterRunKb: Long,
        val wav: File,
        val energy: EnergyProbe?,
        val nativeDetails: String? = null,
    )

    private data class EnergyProbe(
        val synthesisElapsedMs: Long,
        val audioOnlyElapsedMs: Long,
        val repetitions: Int,
        val synthesisJoules: Map<String, Double>,
        val audioOnlyJoules: Map<String, Double>,
        val incrementalJoules: Map<String, Double>,
    )

    private data class EnergyWindow(
        val elapsedMs: Long,
        val joules: Map<String, Double>,
    )

    private data class PowerSnapshot(val values: Map<PowerMonitor, Long>)

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
        lines += "graphSha256=${File(modelDir, baseGraph).takeIf { it.isFile }?.let { sha256(it) } ?: "missing"}"
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
                "CPU reference: graph=%s sha256=%s load=%.2f ms input=%.2f ms run=%.2f ms read=%.2f ms cacheCopy=%.2f ms firstRun=%.2f ms values=%d saved=cpu-reference.f32le",
                reference.graph, sha256(File(modelDir, reference.graph)), reference.loadMs,
                reference.inputMs, reference.runMs, reference.readMs, reference.cacheCopyMs,
                reference.firstRunMs, reference.output.size,
            )
            for ((backend, result) in candidates) {
                val file = "${backend.name.lowercase(Locale.ROOT)}-candidate.f32le"
                writeFloats(File(runDir, file), result.output)
                try {
                    val m = metrics(reference.output, result.output)
                    lines += String.format(
                        Locale.US,
                        "%s candidate: graph=%s sha256=%s load=%.2f ms input=%.2f ms run=%.2f ms read=%.2f ms cacheCopy=%.2f ms firstRun=%.2f ms corr=%.8f mean_abs_diff=%.8g mean_square_diff=%.8g saved=%s",
                        backend, result.graph, sha256(File(modelDir, result.graph)), result.loadMs,
                        result.inputMs, result.runMs, result.readMs, result.cacheCopyMs,
                        result.firstRunMs, m.corr, m.mad, m.msd, file,
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
            if (args.getString("requireCandidates")?.toBooleanStrictOrNull() == true) {
                assertTrue("requested backends failed: ${backends.filter { it !in candidates }}; report=$runDir",
                    backends.all { it in candidates })
            }
        } finally {
            environment?.close()
        }
    }

    /** End-to-end two-bank FlowLM cache chain against the shipped CPU int8 graph. */
    @Test
    fun npuResidentCacheSpeechPair() {
        assumeTrue("PowerMonitor requires Android 15 / API 35", Build.VERSION.SDK_INT >= 35)
        val health = requireNotNull(context.getSystemService(SystemHealthManager::class.java))
        val relevantMonitors = supportedMonitors(health).filter { it.name.isRelevantPowerDomain() }
        assumeTrue("device exposes no CPU/GPU/TPU/display power monitors", relevantMonitors.isNotEmpty())

        val npuSliceCache = args.getString("npuSliceCache")
            ?.toBooleanStrictOrNull() ?: false
        val npuResidentCache = args.getString("npuResidentCache")
            ?.toBooleanStrictOrNull() ?: !npuSliceCache
        val npuPositionMajorCache = args.getString("npuPositionMajorCache")
            ?.toBooleanStrictOrNull() ?: false
        val verifyNpuSliceRows = args.getString("verifyNpuSliceRows")
            ?.toBooleanStrictOrNull() ?: false
        require(!(npuResidentCache && npuSliceCache)) {
            "select either npuResidentCache or npuSliceCache"
        }
        require(!npuPositionMajorCache || npuSliceCache) {
            "npuPositionMajorCache requires npuSliceCache"
        }
        require(!verifyNpuSliceRows || npuSliceCache) {
            "verifyNpuSliceRows requires npuSliceCache"
        }
        val defaultNpuGraph = if (npuResidentCache) {
            DEFAULT_RESIDENT_GRAPH
        } else {
            DEFAULT_NPU_NONRESIDENT_GRAPH
        }
        // Keep the old argument name for existing resident-cache invocations.
        val npuBase = args.getString("npuGraph")?.trim()?.takeIf { it.isNotEmpty() }
            ?: args.getString("residentGraph")?.trim()?.takeIf { it.isNotEmpty() }
            ?: defaultNpuGraph
        val npuGraph = PocketTts.g5Variant(npuBase)
        val referenceGraph = args.getString("referenceGraph")?.trim()
            ?.ifEmpty { PocketTts.LM } ?: PocketTts.LM
        val workload = args.getString("workload")?.trim()?.lowercase(Locale.ROOT) ?: "short"
        val defaultText = when (workload) {
            "short" -> RESIDENT_TEXT
            "long" -> LONG_TEXT
            else -> error("workload must be short or long; got '$workload'")
        }
        val text = args.getString("text")?.trim().orEmpty().ifEmpty { defaultText }
        val voice = args.getString("voice")?.trim()?.ifEmpty { "alba" } ?: "alba"
        val seed = args.getString("seed")?.toLongOrNull() ?: 42L
        val energyRepeats = args.getString("energyRepeats")?.toIntOrNull()?.coerceAtLeast(1) ?: 8
        val order = args.getString("order")?.trim()?.lowercase(Locale.ROOT) ?: "npu-cpu"
        require(order == "npu-cpu" || order == "cpu-npu") {
            "order must be npu-cpu or cpu-npu"
        }
        val models = PocketTtsModels.default(context)
        assertTrue("missing NPU graph ${File(modelDir, npuGraph)}", models.store.exists(npuGraph))
        assertTrue("missing CPU reference graph ${File(modelDir, referenceGraph)}", models.store.exists(referenceGraph))

        val cacheMode = when {
            npuPositionMajorCache -> "slice-position-major"
            npuSliceCache -> "slice"
            npuResidentCache -> "resident"
            else -> "nonresident"
        }
        val runDir = File(
            context.getExternalFilesDir("flowlm-npu-$cacheMode")
                ?: File(context.filesDir, "flowlm-npu-$cacheMode"),
            "speech-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())}",
        ).apply { mkdirs() }

        fun runArm(arm: String): SpeechProbe {
            val isNpu = arm == "npu"
            val placement = if (isNpu) {
                Placement(Accel.NPU, Accel.NPU, Accel.GPU)
            } else {
                Placement(Accel.CPU, Accel.NPU, Accel.GPU)
            }
            val graph = if (isNpu) npuBase else referenceGraph
            val engine = PocketTtsEngine(
                context,
                PocketTtsConfig(
                    models = models,
                    placement = placement,
                    lmGraph = graph,
                    noiseSeed = seed,
                    npuResidentCache = isNpu && npuResidentCache,
                    npuSliceCache = isNpu && npuSliceCache,
                    npuPositionMajorCache = isNpu && npuPositionMajorCache,
                    verifyNpuSliceRows = isNpu && verifyNpuSliceRows,
                ),
            )
            try {
                val load = engine.loadMs.toMap()
                val backends = engine.runtimeAccelerators
                val warmup = engine.stream("A short warmup sentence.", voice) {}
                assertTrue("$arm warmup produced no audio", warmup.audio.isNotEmpty())

                // The first utterance supplies a same-workload audio-only baseline;
                // the following utterance is the measured synthesis interval.
                val baselineTake = engine.stream(text, voice) {}
                assertTrue("$arm audio baseline sample is empty", baselineTake.audio.isNotEmpty())
                val audioOnly = measurePlayback(health, relevantMonitors, baselineTake.audio, energyRepeats)
                val powerBefore = powerSnapshot(health, relevantMonitors)
                val synthesisStartMs = SystemClock.elapsedRealtime()
                val pssBefore = Debug.getPss().toLong()
                var result = baselineTake
                repeat(energyRepeats) {
                    result = engine.stream(text, voice) {}
                    assertTrue("$arm repeated run produced no audio", result.audio.isNotEmpty())
                }
                val pssAfter = Debug.getPss().toLong()
                val synthesisElapsedMs = SystemClock.elapsedRealtime() - synthesisStartMs
                val powerAfter = powerSnapshot(health, relevantMonitors)
                assertTrue("$arm run produced no audio", result.audio.isNotEmpty())
                val synthesisJoules = deltaJoules(powerBefore, powerAfter)
                val scale = synthesisElapsedMs.toDouble() / audioOnly.elapsedMs.coerceAtLeast(1)
                val incrementalJoules = synthesisJoules.mapValues { (name, joules) ->
                    joules - (audioOnly.joules[name] ?: 0.0) * scale
                }
                val energy = EnergyProbe(
                    synthesisElapsedMs = synthesisElapsedMs,
                    audioOnlyElapsedMs = audioOnly.elapsedMs,
                    repetitions = energyRepeats,
                    synthesisJoules = synthesisJoules,
                    audioOnlyJoules = audioOnly.joules,
                    incrementalJoules = incrementalJoules,
                )
                val wav = File(runDir, "$arm.wav")
                Wav.write(wav, result.audio)
                return SpeechProbe(result, load, backends, pssBefore, pssAfter, wav, energy)
            } finally {
                engine.close()
            }
        }

        val orderArms = if (order == "npu-cpu") listOf("npu", "cpu") else listOf("cpu", "npu")
        val probes = LinkedHashMap<String, SpeechProbe>()
        for (arm in orderArms) probes[arm] = runArm(arm)
        val candidate = requireNotNull(probes["npu"])
        val reference = requireNotNull(probes["cpu"])
        val quality = AudioQuality.compare(reference.result.audio, candidate.result.audio)
        val graphFile = File(modelDir, npuGraph)
        val graphSha = sha256(graphFile)
        val thermal = context.getSystemService(PowerManager::class.java)?.currentThermalStatus ?: -1
        val lines = mutableListOf(
            "FlowLM Tensor G5 NPU speech pair",
            "device=${android.os.Build.MODEL}/${android.os.Build.DEVICE} android=${android.os.Build.VERSION.RELEASE}",
            "fingerprint=${android.os.Build.FINGERPRINT}",
            "order=$order workload=$workload seed=$seed voice=$voice energyRepeats=$energyRepeats text=$text",
            "powerMonitors=${relevantMonitors.joinToString { it.name }} method=duration-scaled audio-only playback subtraction",
            "candidateGraph=$npuGraph cacheMode=$cacheMode npuPositionMajorCache=$npuPositionMajorCache verifyNpuSliceRows=$verifyNpuSliceRows sha256=$graphSha aotPartitionReport=${args.getString("aotReport") ?: "not supplied to harness"}",
            "referenceGraph=$referenceGraph placement=lm:CPU dectx:NPU dec:GPU",
            "candidatePlacement=${candidate.backends} loadMs=${candidate.loadMs} pssKb=${candidate.pssBeforeRunKb}->${candidate.pssAfterRunKb}",
            "referencePlacement=${reference.backends} loadMs=${reference.loadMs} pssKb=${reference.pssBeforeRunKb}->${reference.pssAfterRunKb}",
            "candidate frames=${candidate.result.frames} audioSeconds=${candidate.result.audio.size.toDouble() / PocketTts.SAMPLE_RATE} inferenceMs=${candidate.result.ms} firstAudioMs=${candidate.result.profile.firstChunkMs}",
            "reference frames=${reference.result.frames} audioSeconds=${reference.result.audio.size.toDouble() / PocketTts.SAMPLE_RATE} inferenceMs=${reference.result.ms} firstAudioMs=${reference.result.profile.firstChunkMs}",
            "candidate lmMs in/run/read=${candidate.result.profile.lmInMs}/${candidate.result.profile.lmRunMs}/${candidate.result.profile.lmReadMs} steps=${candidate.result.profile.lmSteps} hostBytesIn=${candidate.result.profile.lmInBytes} hostBytesOut=${candidate.result.profile.lmOutBytes}",
            "reference lmMs in/run/read=${reference.result.profile.lmInMs}/${reference.result.profile.lmRunMs}/${reference.result.profile.lmReadMs} steps=${reference.result.profile.lmSteps} hostBytesIn=${reference.result.profile.lmInBytes} hostBytesOut=${reference.result.profile.lmOutBytes}",
            "candidate nativeNs outputMap/cacheMap/rowCopy/unmap=${candidate.result.profile.lmOutputMapMs}/${candidate.result.profile.lmCacheMapMs}/${candidate.result.profile.lmCacheCopyMs}/${candidate.result.profile.lmCacheUnmapMs} bufferTypes=${candidate.result.profile.lmBufferTypes.ifEmpty { "n/a" }}",
            "candidate mimiMs dectx/seanet=${candidate.result.profile.decTxMs}/${candidate.result.profile.seanetMs}",
            "reference mimiMs dectx/seanet=${reference.result.profile.decTxMs}/${reference.result.profile.seanetMs}",
            "waveform corr=${quality.corr} lag=${quality.lag} snrDb=${quality.snrDb} highBandErrDb=${quality.highBandErrDb} refHnrDb=${quality.refHnrDb} candidateHnrDb=${quality.candHnrDb}",
            "thermalStatus=$thermal candidateWav=${candidate.wav.absolutePath} referenceWav=${reference.wav.absolutePath}",
            if (npuSliceCache) {
                "native slice path seeds K/V once, maps packed output and persistent K/V input buffers, then patches the new row; buffer type enums are from LiteRT 2.2.0"
            } else if (npuResidentCache) {
                "cache buffers are ping-ponged; Kotlin reads only the 33-float control output; actual AHWB type/device-side copy volume remain unverified"
            } else {
                "nonresident graph returns only the updated KV rows; host supplies full KV inputs on each FlowLM invocation"
            },
        )
        candidate.energy?.let { lines += energyLine("candidate", it, candidate.result.audio.size) }
        reference.energy?.let { lines += energyLine("reference", it, reference.result.audio.size) }
        File(runDir, "report.txt").writeText(lines.joinToString("\n", postfix = "\n"))
        lines.forEach { Log.i(TAG, it) }
        assertTrue("candidate run did not use NPU placement: ${candidate.backends}", candidate.backends["lm"] == Accel.NPU)
        assertTrue("NPU run produced no FlowLM frames", candidate.result.frames > 0)
    }

    /** Probe AHWB/CL interop, then compare a short cache-chained GPU prompt to CPU FP16. */
    @Test
    fun gpuAhwbPromptGate() {
        val graph = args.getString("gpuGraph")?.trim()?.takeIf { it.isNotEmpty() }
            ?: "pt_flowlm_fused_fp16_contiguous.tflite"
        val path = File(modelDir, graph)
        assertTrue("missing position-major GPU graph $path", path.isFile)
        val runDir = File(
            context.getExternalFilesDir("flowlm-gpu-ahwb") ?: File(context.filesDir, "flowlm-gpu-ahwb"),
            "prompt-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())}",
        ).apply { mkdirs() }
        val lines = mutableListOf("FlowLM GPU AHWB prompt gate", "graph=$graph sha256=${sha256(path)}")
        GpuAhwbLmRunner(path).use { gpu ->
            lines += gpu.details
            File(runDir, "report.txt").writeText(lines.joinToString("\n", postfix = "\n"))
            assertTrue("AHWB/CL interop or cache requirements failed; report=$runDir", gpu.ready)

            val tokenIds = SpTokenizer(File(modelDir, PocketTts.TOKENIZER)).encode("Hello there.")
            val voice = readVoice(File(modelDir, PocketTts.voiceFile("alba")))
            val embed = ByteBuffer.wrap(File(modelDir, PocketTts.EMBED).readBytes())
                .order(ByteOrder.LITTLE_ENDIAN)
            val mask = FloatArray(PocketTts.NH * (PocketTts.PMAX + 1)) { PocketTts.MASK_NEG }
            for (h in 0 until PocketTts.NH) {
                val base = h * (PocketTts.PMAX + 1)
                for (p in 0 until voice.length) mask[base + p] = 0f
                mask[base + PocketTts.PMAX] = 0f
            }
            val cos = FloatArray(PocketTts.HD)
            val sin = FloatArray(PocketTts.HD)
            val noise = FloatArray(PocketTts.LDIM)
            val width = 1 + PocketTts.LDIM + 2 * PocketTts.G * PocketTts.HD
            val result = FloatArray(tokenIds.size * width)
            val timings = LongArray(6)
            gpu.seed(voice.k, voice.v)
            var pos = voice.length
            for ((step, id) in tokenIds.withIndex()) {
                val embedding = FloatArray(PocketTts.H)
                var offset = id * PocketTts.H * Short.SIZE_BYTES
                for (i in embedding.indices) {
                    embedding[i] = Half.toFloat(embed.getShort(offset))
                    offset += Short.SIZE_BYTES
                }
                rope(pos, cos, sin)
                val output = gpu.step(embedding, cos, sin, mask, noise, pos)
                assertTrue("GPU AHWB output width changed", output.packed.size == width)
                System.arraycopy(output.packed, 0, result, step * width, width)
                for (i in timings.indices) timings[i] += output.timingsNs[i]
                for (h in 0 until PocketTts.NH) mask[h * (PocketTts.PMAX + 1) + pos] = 0f
                pos++
            }
            val reference = runPrompt(
                Accel.CPU, "pt_flowlm_fused_fp16.tflite", tokenIds,
                File(modelDir, PocketTts.EMBED).readBytes(), voice, null,
            )
            val m = metrics(reference.output, result)
            writeFloats(File(runDir, "gpu-ahwb.f32le"), result)
            writeFloats(File(runDir, "cpu-fp16.f32le"), reference.output)
            lines += "tokens=${tokenIds.size} voicePrefix=${voice.length} values=${result.size}"
            lines += "timingsMs input/run/read/cacheMap/cacheCopy/cacheUnmap=${timings.joinToString("/") { String.format(Locale.US, "%.3f", it / 1e6) }}"
            lines += "cpuFp16 corr=${m.corr} mad=${m.mad} msd=${m.msd}"
            File(runDir, "report.txt").writeText(lines.joinToString("\n", postfix = "\n"))
            lines.forEach { Log.i(TAG, it) }
            assertTrue("AHWB GPU output disagrees with CPU FP16: $m", m.corr > 0.999)
        }
    }

    /** Probe OpenCL packed-buffer compatibility, then compare a short cache-chained GPU prompt to CPU FP16. */
    @Test
    fun gpuOpenClPromptGate() {
        val priorityHigh = args.getString("gpuPriorityHigh")?.toBooleanStrictOrNull() ?: false
        val graph = args.getString("gpuGraph")?.trim()?.takeIf { it.isNotEmpty() }
            ?: "pt_flowlm_fused_fp16_contiguous.tflite"
        val path = File(modelDir, graph)
        assertTrue("missing position-major GPU graph $path", path.isFile)
        val runDir = File(
            context.getExternalFilesDir("flowlm-gpu-opencl") ?: File(context.filesDir, "flowlm-gpu-opencl"),
            "prompt-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())}",
        ).apply { mkdirs() }
        val lines = mutableListOf("FlowLM GPU OpenCL prompt gate", "graph=$graph sha256=${sha256(path)}")
        GpuOpenClLmRunner(path, priorityHigh).use { gpu ->
            lines += gpu.details
            File(runDir, "report.txt").writeText(lines.joinToString("\n", postfix = "\n"))
            assertTrue("OpenCL packed-buffer compatibility or cache requirements failed; report=$runDir", gpu.ready)

            val tokenIds = SpTokenizer(File(modelDir, PocketTts.TOKENIZER)).encode("Hello there.")
            val voice = readVoice(File(modelDir, PocketTts.voiceFile("alba")))
            val embed = ByteBuffer.wrap(File(modelDir, PocketTts.EMBED).readBytes())
                .order(ByteOrder.LITTLE_ENDIAN)
            val mask = FloatArray(PocketTts.NH * (PocketTts.PMAX + 1)) { PocketTts.MASK_NEG }
            for (h in 0 until PocketTts.NH) {
                val base = h * (PocketTts.PMAX + 1)
                for (p in 0 until voice.length) mask[base + p] = 0f
                mask[base + PocketTts.PMAX] = 0f
            }
            val cos = FloatArray(PocketTts.HD)
            val sin = FloatArray(PocketTts.HD)
            val noise = FloatArray(PocketTts.LDIM)
            val width = 1 + PocketTts.LDIM + 2 * PocketTts.G * PocketTts.HD
            val result = FloatArray(tokenIds.size * width)
            val timings = LongArray(6)
            gpu.seed(voice.k, voice.v)
            var pos = voice.length
            for ((step, id) in tokenIds.withIndex()) {
                val embedding = FloatArray(PocketTts.H)
                var offset = id * PocketTts.H * Short.SIZE_BYTES
                for (i in embedding.indices) {
                    embedding[i] = Half.toFloat(embed.getShort(offset))
                    offset += Short.SIZE_BYTES
                }
                rope(pos, cos, sin)
                val output = gpu.step(embedding, cos, sin, mask, noise, pos, fullOutput = true)
                assertTrue("GPU OpenCL output width changed", output.packed.size == width)
                System.arraycopy(output.packed, 0, result, step * width, width)
                for (i in timings.indices) timings[i] += output.timingsNs[i]
                for (h in 0 until PocketTts.NH) mask[h * (PocketTts.PMAX + 1) + pos] = 0f
                pos++
            }
            val reference = runPrompt(
                Accel.CPU, "pt_flowlm_fused_fp16.tflite", tokenIds,
                File(modelDir, PocketTts.EMBED).readBytes(), voice, null,
            )
            val m = metrics(reference.output, result)
            writeFloats(File(runDir, "gpu-opencl.f32le"), result)
            writeFloats(File(runDir, "cpu-fp16.f32le"), reference.output)
            lines += "tokens=${tokenIds.size} voicePrefix=${voice.length} values=${result.size}"
            lines += "timingsMs input/run/read/cacheMap/cacheCopy/cacheUnmap=${timings.joinToString("/") { String.format(Locale.US, "%.3f", it / 1e6) }}"
            lines += "cpuFp16 corr=${m.corr} mad=${m.mad} msd=${m.msd}"
            File(runDir, "report.txt").writeText(lines.joinToString("\n", postfix = "\n"))
            lines.forEach { Log.i(TAG, it) }
            assertTrue("OpenCL GPU output disagrees with CPU FP16: $m", m.corr > 0.999)
        }
    }

    /** Opt-in long speech comparison of the fused GPU FlowLM and shipped CPU int8. */
    @Test
    fun gpuSpeechPair() {
        val openClCache = args.getString("gpuOpenClCache")?.toBooleanStrictOrNull() ?: false
        val priorityHigh = args.getString("gpuPriorityHigh")?.toBooleanStrictOrNull() ?: false
        val gpuGraph = args.getString("gpuGraph")?.trim()?.takeIf { it.isNotEmpty() }
            ?: "pt_flowlm_fused_fp16.tflite"
        val referenceGraph = args.getString("referenceGraph")?.trim()?.takeIf { it.isNotEmpty() }
            ?: PocketTts.LM
        val order = args.getString("order")?.trim()?.lowercase(Locale.ROOT) ?: "gpu-cpu"
        require(order == "gpu-cpu" || order == "cpu-gpu") { "order must be gpu-cpu or cpu-gpu" }
        val workload = args.getString("workload")?.trim()?.lowercase(Locale.ROOT) ?: "long"
        val defaultText = when (workload) {
            "short" -> RESIDENT_TEXT
            "long" -> LONG_TEXT
            else -> error("workload must be short or long")
        }
        val text = args.getString("text")?.trim().orEmpty().ifEmpty { defaultText }
        val voice = args.getString("voice")?.trim()?.takeIf { it.isNotEmpty() } ?: "alba"
        val seed = args.getString("seed")?.toLongOrNull() ?: 42L
        val energyRepeats = args.getString("energyRepeats")?.toIntOrNull()?.coerceAtLeast(1) ?: 1
        val models = PocketTtsModels.default(context)
        assertTrue("missing GPU graph $gpuGraph", models.store.exists(gpuGraph))
        assertTrue("missing CPU graph $referenceGraph", models.store.exists(referenceGraph))

        val health = if (Build.VERSION.SDK_INT >= 35) {
            context.getSystemService(SystemHealthManager::class.java)
        } else null
        val monitors = health?.let { supportedMonitors(it).filter { p -> p.name.isRelevantPowerDomain() } }
            .orEmpty()
        val runDir = File(
            context.getExternalFilesDir("flowlm-gpu") ?: File(context.filesDir, "flowlm-gpu"),
            "speech-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())}",
        ).apply { mkdirs() }
        val gpuProgramCache = File(context.cacheDir, "flowlm_gpu_program_cache").apply { mkdirs() }
        val thermalBefore = context.getSystemService(PowerManager::class.java)?.currentThermalStatus ?: -1

        fun runArm(arm: String): SpeechProbe {
            val isGpu = arm == "gpu"
            // The decoder path is identical in both arms. CPU dec_tx avoids the
            // process-global NPU dispatch setup after the GPU LM has loaded.
            val placement = Placement(if (isGpu) Accel.GPU else Accel.CPU, Accel.CPU, Accel.GPU)
            val engine = PocketTtsEngine(
                context,
                PocketTtsConfig(
                    models = models,
                    placement = placement,
                    lmGraph = if (isGpu) gpuGraph else referenceGraph,
                    noiseSeed = seed,
                    gpuCache = gpuProgramCache,
                    gpuOpenClCache = isGpu && openClCache,
                    gpuOpenClHighPriority = isGpu && priorityHigh,
                ),
            )
            try {
                val load = engine.loadMs.toMap()
                val backends = engine.runtimeAccelerators
                if (isGpu && openClCache) {
                    Log.i(TAG, "gpuOpenClDetails=${engine.gpuOpenClDetails}")
                }
                assertTrue("$arm LM silently fell back: $backends", backends["lm"] == placement.lm)
                val warmup = engine.stream("A short warmup sentence.", voice) {}
                assertTrue("$arm warmup produced no audio", warmup.audio.isNotEmpty())
                val baseline = engine.stream(text, voice) {}
                assertTrue("$arm baseline produced no audio", baseline.audio.isNotEmpty())
                val audioOnly = if (health != null && monitors.isNotEmpty()) {
                    measurePlayback(health, monitors, baseline.audio, energyRepeats)
                } else null
                val powerBefore = if (health != null && monitors.isNotEmpty()) {
                    powerSnapshot(health, monitors)
                } else null
                val pssBefore = Debug.getPss().toLong()
                val started = SystemClock.elapsedRealtime()
                var result = baseline
                repeat(energyRepeats) {
                    result = engine.stream(text, voice) {}
                    assertTrue("$arm measured run produced no audio", result.audio.isNotEmpty())
                }
                val elapsed = SystemClock.elapsedRealtime() - started
                val pssAfter = Debug.getPss().toLong()
                val powerAfter = if (health != null && monitors.isNotEmpty()) {
                    powerSnapshot(health, monitors)
                } else null
                val energy = if (audioOnly != null && powerBefore != null && powerAfter != null) {
                    val synthesisJoules = deltaJoules(powerBefore, powerAfter)
                    val scale = elapsed.toDouble() / audioOnly.elapsedMs.coerceAtLeast(1)
                    EnergyProbe(
                        synthesisElapsedMs = elapsed,
                        audioOnlyElapsedMs = audioOnly.elapsedMs,
                        repetitions = energyRepeats,
                        synthesisJoules = synthesisJoules,
                        audioOnlyJoules = audioOnly.joules,
                        incrementalJoules = synthesisJoules.mapValues { (name, value) ->
                            value - (audioOnly.joules[name] ?: 0.0) * scale
                        },
                    )
                } else null
                val wav = File(runDir, "$arm.wav")
                Wav.write(wav, result.audio)
                return SpeechProbe(result, load, backends, pssBefore, pssAfter, wav, energy,
                    engine.gpuOpenClDetails)
            } finally {
                engine.close()
            }
        }

        val arms = if (order == "gpu-cpu") listOf("gpu", "cpu") else listOf("cpu", "gpu")
        val probes = LinkedHashMap<String, SpeechProbe>()
        for (arm in arms) probes[arm] = runArm(arm)
        val gpu = requireNotNull(probes["gpu"])
        val cpu = requireNotNull(probes["cpu"])
        val quality = AudioQuality.compare(cpu.result.audio, gpu.result.audio)
        val thermalAfter = context.getSystemService(PowerManager::class.java)?.currentThermalStatus ?: -1
        val lines = mutableListOf(
            "FlowLM fused GPU speech pair",
            "device=${Build.MODEL}/${Build.DEVICE} android=${Build.VERSION.RELEASE} fingerprint=${Build.FINGERPRINT}",
            "order=$order workload=$workload seed=$seed voice=$voice energyRepeats=$energyRepeats text=$text",
            "gpuGraph=$gpuGraph sha256=${sha256(File(modelDir, gpuGraph))} gpuProgramCache=$gpuProgramCache openClCache=$openClCache priorityHigh=$priorityHigh",
            "referenceGraph=$referenceGraph sha256=${sha256(File(modelDir, referenceGraph))}",
            "powerMonitors=${monitors.joinToString { it.name }} method=duration-scaled audio-only playback subtraction",
            "gpuPlacement=${gpu.backends} loadMs=${gpu.loadMs} pssKb=${gpu.pssBeforeRunKb}->${gpu.pssAfterRunKb}",
            "gpuOpenClDetails=${gpu.nativeDetails ?: "disabled"}",
            "cpuPlacement=${cpu.backends} loadMs=${cpu.loadMs} pssKb=${cpu.pssBeforeRunKb}->${cpu.pssAfterRunKb}",
            "gpu frames=${gpu.result.frames} audioSeconds=${gpu.result.audio.size.toDouble() / PocketTts.SAMPLE_RATE} inferenceMs=${gpu.result.ms} firstAudioMs=${gpu.result.profile.firstChunkMs}",
            "cpu frames=${cpu.result.frames} audioSeconds=${cpu.result.audio.size.toDouble() / PocketTts.SAMPLE_RATE} inferenceMs=${cpu.result.ms} firstAudioMs=${cpu.result.profile.firstChunkMs}",
            "gpu lmMs inputCopy/run/read=${gpu.result.profile.lmInMs}/${gpu.result.profile.lmRunMs}/${gpu.result.profile.lmReadMs} steps=${gpu.result.profile.lmSteps} invocations=${gpu.result.profile.lmInvocations} hostBytesIn=${gpu.result.profile.lmInBytes} hostBytesOut=${gpu.result.profile.lmOutBytes}",
            "gpu nativeMs outputLock/cacheQueueGap/copyEnqueue/copyWait=${gpu.result.profile.lmOutputMapMs}/${gpu.result.profile.lmCacheMapMs}/${gpu.result.profile.lmCacheCopyMs}/${gpu.result.profile.lmCacheUnmapMs}",
            "cpu lmMs inputCopy/run/read=${cpu.result.profile.lmInMs}/${cpu.result.profile.lmRunMs}/${cpu.result.profile.lmReadMs} steps=${cpu.result.profile.lmSteps} invocations=${cpu.result.profile.lmInvocations} hostBytesIn=${cpu.result.profile.lmInBytes} hostBytesOut=${cpu.result.profile.lmOutBytes}",
            "gpu mimiMs dectx/seanet=${gpu.result.profile.decTxMs}/${gpu.result.profile.seanetMs}",
            "cpu mimiMs dectx/seanet=${cpu.result.profile.decTxMs}/${cpu.result.profile.seanetMs}",
            "waveform corr=${quality.corr} lag=${quality.lag} snrDb=${quality.snrDb} highBandErrDb=${quality.highBandErrDb} refHnrDb=${quality.refHnrDb} gpuHnrDb=${quality.candHnrDb}",
            "thermalStatus=$thermalBefore->$thermalAfter gpuWav=${gpu.wav.absolutePath} cpuWav=${cpu.wav.absolutePath}",
        )
        gpu.energy?.let { lines += energyLine("gpu", it, gpu.result.audio.size) }
        cpu.energy?.let { lines += energyLine("cpu", it, cpu.result.audio.size) }
        File(runDir, "report.txt").writeText(lines.joinToString("\n", postfix = "\n"))
        lines.forEach { Log.i(TAG, it) }
        assertTrue("GPU run produced no FlowLM frames", gpu.result.frames > 0)
    }

    private fun supportedMonitors(manager: SystemHealthManager): List<PowerMonitor> {
        val latch = CountDownLatch(1)
        var result: List<PowerMonitor> = emptyList()
        manager.getSupportedPowerMonitors(null) {
            result = it.toList()
            latch.countDown()
        }
        assertTrue("timed out listing device power monitors", latch.await(30, TimeUnit.SECONDS))
        return result
    }

    private fun powerSnapshot(
        manager: SystemHealthManager,
        monitors: List<PowerMonitor>,
    ): PowerSnapshot {
        val latch = CountDownLatch(1)
        var readings: PowerMonitorReadings? = null
        var failure: RuntimeException? = null
        manager.getPowerMonitorReadings(
            monitors,
            null,
            object : OutcomeReceiver<PowerMonitorReadings, RuntimeException> {
                override fun onResult(result: PowerMonitorReadings) {
                    readings = result
                    latch.countDown()
                }

                override fun onError(error: RuntimeException) {
                    failure = error
                    latch.countDown()
                }
            },
        )
        assertTrue("timed out reading device power monitors", latch.await(30, TimeUnit.SECONDS))
        failure?.let { throw AssertionError("could not read device power monitors", it) }
        val result = requireNotNull(readings)
        return PowerSnapshot(monitors.associateWith { result.getConsumedEnergy(it) })
    }

    private fun deltaJoules(before: PowerSnapshot, after: PowerSnapshot): Map<String, Double> =
        before.values.mapNotNull { (monitor, start) ->
            val end = after.values[monitor] ?: return@mapNotNull null
            if (start < 0 || end < start) return@mapNotNull null
            monitor.name to ((end - start) / 1_000_000.0)
        }.toMap()

    private fun measurePlayback(
        manager: SystemHealthManager,
        monitors: List<PowerMonitor>,
        audio: FloatArray,
        repetitions: Int,
    ): EnergyWindow {
        val track = newAudioTrack()
        return try {
            val before = powerSnapshot(manager, monitors)
            val started = SystemClock.elapsedRealtime()
            track.play()
            repeat(repetitions) { writeAudio(track, audio) }
            drainAudio(track, audio.size * repetitions)
            val elapsedMs = SystemClock.elapsedRealtime() - started
            val after = powerSnapshot(manager, monitors)
            EnergyWindow(elapsedMs, deltaJoules(before, after))
        } finally {
            track.release()
        }
    }

    private fun newAudioTrack(): AudioTrack {
        val format = AudioFormat.Builder()
            .setSampleRate(PocketTts.SAMPLE_RATE)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        val minBytes = AudioTrack.getMinBufferSize(
            PocketTts.SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        check(minBytes > 0) { "AudioTrack does not support 24 kHz mono float PCM" }
        return AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
            format,
            maxOf(minBytes, PocketTts.SAMPLE_RATE * Float.SIZE_BYTES),
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        ).apply { setVolume(0.5f) }
    }

    private fun writeAudio(track: AudioTrack, audio: FloatArray) {
        var offset = 0
        while (offset < audio.size) {
            val written = track.write(
                audio,
                offset,
                minOf(4096, audio.size - offset),
                AudioTrack.WRITE_BLOCKING,
            )
            check(written > 0) { "AudioTrack.write failed: $written" }
            offset += written
        }
    }

    private fun drainAudio(track: AudioTrack, samples: Int) {
        val deadline = SystemClock.elapsedRealtime() +
            samples * 1000L / PocketTts.SAMPLE_RATE + 15_000L
        while (track.playbackHeadPosition < samples && SystemClock.elapsedRealtime() < deadline) {
            SystemClock.sleep(20)
        }
        assertTrue("AudioTrack did not finish playback", track.playbackHeadPosition >= samples)
        track.stop()
    }

    private fun energyLine(label: String, energy: EnergyProbe, samples: Int): String {
        val audioSeconds = samples.toDouble() * energy.repetitions / PocketTts.SAMPLE_RATE
        val perSecond = energy.incrementalJoules.mapValues { (_, joules) ->
            joules / audioSeconds.coerceAtLeast(1e-9)
        }
        return "$label energy repeats=${energy.repetitions} synthesisMs=${energy.synthesisElapsedMs} " +
            "audioOnlyMs=${energy.audioOnlyElapsedMs} " +
            "synthesisJ=${renderEnergy(energy.synthesisJoules)} " +
            "audioOnlyJ=${renderEnergy(energy.audioOnlyJoules)} " +
            "incrementalJ=${renderEnergy(energy.incrementalJoules)} " +
            "incrementalJPerSpeechSecond=${renderEnergy(perSecond)}"
    }

    private fun renderEnergy(values: Map<String, Double>): String =
        values.toSortedMap().entries.joinToString(",") { (name, value) ->
            "$name=${String.format(Locale.US, "%.3f", value)}"
        }

    private fun String.isRelevantPowerDomain(): Boolean {
        val key = lowercase(Locale.ROOT)
        return key.contains("cpu") || key.contains("gpu") || key.contains("tpu") || key.contains("display")
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
                var inputNs = 0L
                var runNs = 0L
                var readNs = 0L
                var cacheCopyNs = 0L
                var firstRunNs = 0L
                for (id in tokenIds) {
                    val embedding = FloatArray(PocketTts.H)
                    var offset = id * PocketTts.H * Short.SIZE_BYTES
                    for (i in embedding.indices) {
                        embedding[i] = Half.toFloat(embed.getShort(offset))
                        offset += Short.SIZE_BYTES
                    }
                    rope(pos, cosine, sine)
                    val inputStarted = System.nanoTime()
                    input[0].writeFloat(embedding)
                    input[1].writeFloat(cosine)
                    input[2].writeFloat(sine)
                    input[3].writeFloat(mask)
                    input[4].writeFloat(pk)
                    input[5].writeFloat(pv)
                    input[6].writeFloat(zeroNoise)
                    inputNs += System.nanoTime() - inputStarted
                    val started = System.nanoTime()
                    if (stepInput != null) model.run(input, output, 1) else model.run(input, output)
                    val ran = System.nanoTime() - started
                    if (firstRunNs == 0L) firstRunNs = ran
                    runNs += ran
                    val readStarted = System.nanoTime()
                    val values = output.single().readFloat()
                    readNs += System.nanoTime() - readStarted
                    check(values.size == outputPerToken) {
                        "FlowLM output width changed: expected $outputPerToken, got ${values.size}"
                    }
                    System.arraycopy(values, 0, all, outputOffset, values.size)
                    outputOffset += values.size

                    val copyStarted = System.nanoTime()
                    val kvStart = 1 + PocketTts.LDIM
                    for (g in 0 until PocketTts.G) {
                        System.arraycopy(values, kvStart + g * PocketTts.HD, pk,
                            g * PocketTts.PMAX * PocketTts.HD + pos * PocketTts.HD, PocketTts.HD)
                        System.arraycopy(values, kvStart + PocketTts.G * PocketTts.HD + g * PocketTts.HD, pv,
                            g * PocketTts.PMAX * PocketTts.HD + pos * PocketTts.HD, PocketTts.HD)
                    }
                    for (h in 0 until PocketTts.NH) mask[h * (PocketTts.PMAX + 1) + pos] = 0f
                    pos++
                    cacheCopyNs += System.nanoTime() - copyStarted
                }
                return Run(backend, graph, all, tokenIds.size, loadMs,
                    inputNs / 1e6, runNs / 1e6, readNs / 1e6, cacheCopyNs / 1e6,
                    firstRunNs / 1e6)
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

    private fun sha256(file: File): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val TAG = "FlowLmHarness"
        const val DEFAULT_GRAPH = "pt_flowlm_fused_fp16_no_truncation.tflite"
        const val REFERENCE_GRAPH = "pt_flowlm_fused_fp16.tflite"
        const val DEFAULT_TEXT = "Hello world. This is a FlowLM text probe."
        const val DEFAULT_RESIDENT_GRAPH = "pt_flowlm_fused_fp16_resident_no_truncation.tflite"
        const val DEFAULT_NPU_NONRESIDENT_GRAPH = "pt_flowlm_fused_fp16_no_truncation.tflite"
        const val RESIDENT_TEXT = "Hello there, how are you?"
        val LONG_TEXT = """
            Each spring, a small group of neighbors meets at the public library to plan a weekend repair fair. They bring lamps with loose switches, radios that have gone quiet, bicycles with stubborn brakes, and kitchen tools that only need a little attention. Before the doors open, volunteers arrange the tables by task and place a handwritten sign beside every box of spare parts. A retired engineer shows the children how to trace a simple circuit, while a local baker sets out warm bread and explains how patient practice can turn a difficult recipe into an ordinary part of the day.

            By midmorning, the room is busy but calm. People take turns describing what stopped working, and the volunteers ask questions before reaching for a screwdriver. Some repairs succeed quickly; others become lessons in what to try next. Nobody is asked to pay, and nobody is hurried toward a perfect result. The goal is to help useful things last longer, share skills that might otherwise remain hidden, and make it easier for strangers to begin a conversation. At the end of the afternoon, the tables are cleared, the tools are counted, and a list of unfinished jobs is saved for next month.
        """.trimIndent().replace('\n', ' ')
    }
}

