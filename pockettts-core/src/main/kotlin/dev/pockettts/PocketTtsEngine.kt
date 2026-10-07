package dev.pockettts

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.LinkedHashMap
import java.util.Arrays
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future

internal data class ResidentLmStepRun(
    val control: FloatArray,
    val inputNs: Long,
    val runNs: Long,
    val readNs: Long,
    val outputMapNs: Long = 0,
    val cacheMapNs: Long = 0,
    val cacheCopyNs: Long = 0,
    val cacheUnmapNs: Long = 0,
)

/**
 * Owns the compiled graphs, the host assets and the worker thread. Create one
 * per process (loading costs seconds and ~150 MB of native memory), then get a
 * [PocketTtsSession] per utterance.
 *
 * The engine is safe to use from multiple threads: every buffer-touching call is
 * serialized on an internal lock, and asynchronous utterances run on a single
 * worker so the LiteRT contexts are never invoked concurrently.
 */
class PocketTtsEngine(
    context: Context,
    val config: PocketTtsConfig = PocketTtsConfig.default(context),
) : Closeable {

    private val appCtx: Context = context.applicationContext
    private val models: PocketTtsModels = config.models
    val placement: Placement = config.placement
    val lmSteps: Int = config.lmSteps
    val streamW: Int = config.streamW
    val codecContinuity: Boolean = config.codecContinuity
    val noiseSeed: Long? = config.noiseSeed
    internal val streamDecoderGraphName: String =
        config.streamDecoderGraph ?: PocketTts.deconlyGraph(streamW)

    /** The voices this engine can speak, first = default. */
    val voices: List<Voice> = config.voices
    private val gpuCache: File? = config.gpuCache

    /** Per-graph compile time (ms), keyed lm/lm_ms/dectx/dec/dec_w. */
    val loadMs = LinkedHashMap<String, Long>()

    /** Backend that successfully created each graph; non-NPU failures retry on CPU. */
    private val loadedAccelerators = LinkedHashMap<String, Accel>()
    val runtimeAccelerators: Map<String, Accel> get() = loadedAccelerators.toMap()

    private var npuEnvironment: Environment? = null

    private fun npuEnv(): Environment {
        npuEnvironment?.let { return it }
        val env = Environment.create(
            appCtx,
            mapOf(
                Environment.Option.DispatchLibraryDir to appCtx.applicationInfo.nativeLibraryDir,
            ),
        )
        npuEnvironment = env
        return env
    }

    private fun load(name: String, key: String, accel: Accel): CompiledModel {
        val file = if (accel == Accel.NPU) PocketTts.g5Variant(name) else name
        val p = models.store.file(file).absolutePath
        val t = System.nanoTime()
        var actualAccel = accel
        val model = try {
            when (accel) {
                Accel.CPU -> CompiledModel.create(p, CompiledModel.Options(Accelerator.CPU), null)
                Accel.GPU, Accel.GPU32 -> {
                    val opts = CompiledModel.Options(Accelerator.GPU)
                    val gpu = if (accel == Accel.GPU32) {
                        CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
                    } else {
                        CompiledModel.GpuOptions()
                    }
                    opts.gpuOptions = if (gpuCache != null) {
                        gpu.copy(
                            serializationDir = gpuCache.absolutePath,
                            modelCacheKey = name,
                            serializeProgramCache = true,
                        )
                    } else {
                        gpu
                    }
                    CompiledModel.create(p, opts, null)
                }
                Accel.NPU -> CompiledModel.create(p, CompiledModel.Options(Accelerator.NPU), npuEnv())
            }
        } catch (e: Throwable) {
            // The NPU partition only has a dispatch kernel, so a CPU retry of the
            // same file cannot work; a silent fallback would hide a mis-wired NPU.
            if (accel == Accel.NPU) throw e
            actualAccel = Accel.CPU
            android.util.Log.w(
                "PocketTTS",
                "$key graph $name failed to load on $accel; retrying on CPU",
                e,
            )
            CompiledModel.create(p, CompiledModel.Options(Accelerator.CPU), null)
        }
        loadMs[key] = (System.nanoTime() - t) / 1_000_000
        loadedAccelerators[key] = actualAccel
        return model
    }

    // ---- graphs -----------------------------------------------------------
    val lmGraphName: String = config.lmGraph ?: PocketTts.LM

    internal val lm: CompiledModel = load(lmGraphName, "lm", placement.lm)
    internal val lmMs: CompiledModel? =
        if (lmSteps > 1) load(PocketTts.msGraph(lmSteps), "lm_ms", placement.lm) else null
    internal val dectx: CompiledModel = load(PocketTts.DEC_TX, "dectx", placement.dectx)
    internal val deconly: CompiledModel = load(PocketTts.DECONLY, "dec", placement.deconly)
    internal val deconlyW: CompiledModel? = if (models.store.exists(streamDecoderGraphName)) {
        load(streamDecoderGraphName, "dec_w", placement.deconly)
    } else {
        null
    }

    /**
     * `litert_torch` lays named signatures before the default one, so in a file
     * that carries the prefill the fused step is index 1 and the prefill is
     * index 0. Buffers belong to a signature, so the step's must be created
     * explicitly; a single-signature file (older drops) has only index 0.
     * Created once and kept -- probing with a throwaway set would transiently
     * double the ~25 MB packed-KV inputs.
     */
    private val lmStepIn: List<TensorBuffer>? =
        runCatching { lm.createInputBuffers(1) }.getOrNull()

    internal val lmIn: List<TensorBuffer> = if (config.npuResidentCache) {
        emptyList()
    } else {
        lmStepIn ?: lm.createInputBuffers()
    }
    internal val lmOut: List<TensorBuffer> = if (config.npuResidentCache) {
        emptyList()
    } else if (lmStepIn != null) {
        lm.createOutputBuffers(1)
    } else {
        lm.createOutputBuffers()
    }

    /**
     * The experimental graph has eight inputs and three outputs:
     * x/cos/sin/mask/K/V/noise/write-mask -> control/next-K/next-V. The two
     * model-created cache banks exchange input/output roles after each run.
     * The 33-float control tensor is the only output read by Kotlin.
     */
    internal val usesNpuResidentCache: Boolean = config.npuResidentCache
    internal val usesNpuSliceCache: Boolean = config.npuSliceCache
    internal val usesNpuPositionMajorCache: Boolean = config.npuPositionMajorCache
    internal val usesGpuOpenClCache: Boolean = config.gpuOpenClCache
    private val gpuOpenClRunner: GpuOpenClLmRunner? = if (usesGpuOpenClCache) {
        val started = System.nanoTime()
        GpuOpenClLmRunner(models.store.file(lmGraphName)).also {
            check(it.ready) { "GPU OpenCL packed cache requirements failed: ${it.details}" }
            loadMs["lm_opencl"] = (System.nanoTime() - started) / 1_000_000
            android.util.Log.i("PocketTTSTime", it.details)
        }
    } else null
    val gpuOpenClDetails: String? get() = gpuOpenClRunner?.details
    private val positionMajorSeedK: FloatArray? = if (usesNpuPositionMajorCache) {
        FloatArray(PocketTts.G * PocketTts.PMAX * PocketTts.HD)
    } else null
    private val positionMajorSeedV: FloatArray? = if (usesNpuPositionMajorCache) {
        FloatArray(PocketTts.G * PocketTts.PMAX * PocketTts.HD)
    } else null
    internal val lmResidentIn: MutableList<TensorBuffer>? =
        if (usesNpuResidentCache) lm.createInputBuffers().toMutableList() else null
    internal val lmResidentOut: MutableList<TensorBuffer>? =
        if (usesNpuResidentCache) lm.createOutputBuffers().toMutableList() else null
    private val residentBankAK: TensorBuffer? = lmResidentIn?.getOrNull(4)
    private val residentBankAV: TensorBuffer? = lmResidentIn?.getOrNull(5)
    private val residentBankBK: TensorBuffer? = lmResidentOut?.getOrNull(1)
    private val residentBankBV: TensorBuffer? = lmResidentOut?.getOrNull(2)
    private val residentWriteMask = if (usesNpuResidentCache) FloatArray(PocketTts.PMAX) else null

    /**
     * Prompt prefill is a second signature of [lm], not a separate graph: it
     * shares the backbone's weight buffers, so it costs no extra storage. Null
     * on model drops that predate it, in which case the prompt falls back to a
     * fused step per token.
     */
    internal val prefillIn: List<TensorBuffer>? =
        if (config.usePrefill && lmStepIn != null) {
            runCatching { lm.createInputBuffers(0) }.getOrNull()
        } else null
    internal val prefillOut: List<TensorBuffer>? =
        if (config.usePrefill && lmStepIn != null) {
            runCatching { lm.createOutputBuffers(0) }.getOrNull()
        } else null

    init {
        if (usesNpuResidentCache) {
            check(lmResidentIn?.size == 8) {
                "NPU-resident FlowLM graph must have 8 inputs; got ${lmResidentIn?.size}"
            }
            check(lmResidentOut?.size == 3) {
                "NPU-resident FlowLM graph must have 3 outputs; got ${lmResidentOut?.size}"
            }
            check(
                residentBankAK != null && residentBankAV != null &&
                    residentBankBK != null && residentBankBV != null,
            ) { "NPU-resident FlowLM cache banks are missing" }
        }
        if (usesNpuSliceCache) {
            check(lmIn.size == 7 && lmOut.size == 1) {
                "NPU slice FlowLM expects 7 inputs and one packed output; got ${lmIn.size}/${lmOut.size}"
            }
        }
    }

    /** Actual backing types; 2 is AHardwareBuffer, 1 is host memory. */
    internal val npuSliceBufferTypes: IntArray? = if (usesNpuSliceCache) {
        NpuSliceCacheBridge.bufferTypes(lmIn[4], lmIn[5], lmOut[0]).also {
            android.util.Log.i("PocketTTSTime", "NPU slice buffer types K/V/out=${it.joinToString()}")
        }
    } else null

    /** Run the fused step on buffers created above (last signature when named). */
    internal fun runLm(ins: List<TensorBuffer>, outs: List<TensorBuffer>) {
        if (lmStepIn != null) lm.run(ins, outs, 1) else lm.run(ins, outs)
    }

    /** Seed one cache bank from the selected voice and restore A -> B roles. */
    internal fun resetNpuResidentCache(k: FloatArray, v: FloatArray): Long {
        check(usesNpuResidentCache) { "NPU-resident cache is disabled" }
        val expected = PocketTts.G * PocketTts.PMAX * PocketTts.HD
        require(k.size == expected && v.size == expected) {
            "NPU-resident cache expects $expected floats per bank; got ${k.size}/${v.size}"
        }
        val start = System.nanoTime()
        val inputs = requireNotNull(lmResidentIn)
        val outputs = requireNotNull(lmResidentOut)
        val bankAK = requireNotNull(residentBankAK)
        val bankAV = requireNotNull(residentBankAV)
        val bankBK = requireNotNull(residentBankBK)
        val bankBV = requireNotNull(residentBankBV)
        bankAK.writeFloat(k)
        bankAV.writeFloat(v)
        inputs[4] = bankAK
        inputs[5] = bankAV
        outputs[1] = bankBK
        outputs[2] = bankBV
        return System.nanoTime() - start
    }

    /** Run one frame and chain full K/V outputs without reading cache data. */
    internal fun runNpuResidentLm(
        emb: FloatArray,
        cos: FloatArray,
        sin: FloatArray,
        mask: FloatArray,
        noise: FloatArray,
        position: Int,
    ): ResidentLmStepRun {
        check(usesNpuResidentCache) { "NPU-resident cache is disabled" }
        require(position in 0 until PocketTts.PMAX) { "cache position out of range: $position" }
        val inputs = requireNotNull(lmResidentIn)
        val outputs = requireNotNull(lmResidentOut)
        val writeMask = requireNotNull(residentWriteMask)

        var started = System.nanoTime()
        inputs[0].writeFloat(emb)
        inputs[1].writeFloat(cos)
        inputs[2].writeFloat(sin)
        inputs[3].writeFloat(mask)
        inputs[6].writeFloat(noise)
        Arrays.fill(writeMask, 0f)
        writeMask[position] = 1f
        inputs[7].writeFloat(writeMask)
        val inputNs = System.nanoTime() - started

        started = System.nanoTime()
        lm.run(inputs, outputs, 0)
        val runNs = System.nanoTime() - started

        started = System.nanoTime()
        val control = outputs[0].readFloat()
        val readNs = System.nanoTime() - started
        check(control.size == 1 + PocketTts.LDIM) {
            "NPU-resident FlowLM control output must have ${1 + PocketTts.LDIM} floats; got ${control.size}"
        }

        val oldK = inputs[4]
        val oldV = inputs[5]
        inputs[4] = outputs[1]
        inputs[5] = outputs[2]
        outputs[1] = oldK
        outputs[2] = oldV
        return ResidentLmStepRun(control, inputNs, runNs, readNs)
    }

    /** Seed persistent NPU inputs once per utterance; later updates touch one row. */
    internal fun resetNpuSliceCache(k: FloatArray, v: FloatArray): Long {
        check(usesNpuSliceCache) { "NPU slice cache is disabled" }
        val expected = PocketTts.G * PocketTts.PMAX * PocketTts.HD
        require(k.size == expected && v.size == expected)
        val start = System.nanoTime()
        if (usesNpuPositionMajorCache) {
            fun positionMajor(groupMajor: FloatArray, out: FloatArray) {
                for (p in 0 until PocketTts.PMAX) {
                    for (g in 0 until PocketTts.G) {
                        System.arraycopy(
                            groupMajor, (g * PocketTts.PMAX + p) * PocketTts.HD,
                            out, (p * PocketTts.G + g) * PocketTts.HD,
                            PocketTts.HD,
                        )
                    }
                }
            }
            val seedK = requireNotNull(positionMajorSeedK)
            val seedV = requireNotNull(positionMajorSeedV)
            positionMajor(k, seedK)
            positionMajor(v, seedV)
            lmIn[4].writeFloat(seedK)
            lmIn[5].writeFloat(seedV)
        } else {
            lmIn[4].writeFloat(k)
            lmIn[5].writeFloat(v)
        }
        return System.nanoTime() - start
    }

    /** Uses the existing packed-output G5 graph without a Kotlin K/V round-trip. */
    internal fun runNpuSliceLm(
        emb: FloatArray, cos: FloatArray, sin: FloatArray, mask: FloatArray,
        noise: FloatArray, position: Int,
    ): ResidentLmStepRun {
        check(usesNpuSliceCache) { "NPU slice cache is disabled" }
        require(position in 0 until PocketTts.PMAX)
        var started = System.nanoTime()
        lmIn[0].writeFloat(emb)
        lmIn[1].writeFloat(cos)
        lmIn[2].writeFloat(sin)
        lmIn[3].writeFloat(mask)
        lmIn[6].writeFloat(noise)
        val inputNs = System.nanoTime() - started
        started = System.nanoTime()
        runLm(lmIn, lmOut)
        val runNs = System.nanoTime() - started
        started = System.nanoTime()
        val timings = LongArray(4)
        val control = NpuSliceCacheBridge.update(
            lmIn[4], lmIn[5], lmOut[0], position,
            PocketTts.PMAX, PocketTts.G, PocketTts.HD,
            usesNpuPositionMajorCache, timings,
        )
        if (config.verifyNpuSliceRows) {
            val delta = NpuSliceCacheBridge.rowMaxDifference(
                lmIn[4], lmIn[5], lmOut[0], position,
                PocketTts.PMAX, PocketTts.G, PocketTts.HD,
                usesNpuPositionMajorCache,
            )
            check(delta == 0f) { "NPU slice K/V row mismatch at $position: $delta" }
        }
        val readNs = System.nanoTime() - started
        check(control.size == 1 + PocketTts.LDIM)
        return ResidentLmStepRun(
            control, inputNs, runNs, readNs,
            timings[0], timings[1], timings[2], timings[3],
        )
    }

    /** Native GPU caches are reseeded for every utterance under the engine lock. */
    internal fun resetGpuOpenClCache(k: FloatArray, v: FloatArray): Long {
        val started = System.nanoTime()
        requireNotNull(gpuOpenClRunner).seed(k, v)
        return System.nanoTime() - started
    }

    internal fun runGpuOpenClLm(
        emb: FloatArray, cos: FloatArray, sin: FloatArray, mask: FloatArray,
        noise: FloatArray, position: Int,
    ): ResidentLmStepRun {
        val result = requireNotNull(gpuOpenClRunner).step(emb, cos, sin, mask, noise, position)
        val t = result.timingsNs
        check(result.packed.size == 1 + PocketTts.LDIM)
        return ResidentLmStepRun(
            result.packed, t[0], t[1], t[2] + t[3] + t[4] + t[5],
            t[2], t[3], t[4], t[5],
        )
    }

    internal val lmMsIn = lmMs?.createInputBuffers()
    internal val lmMsOut = lmMs?.createOutputBuffers()
    internal val dectxIn = dectx.createInputBuffers()
    internal val dectxOut = dectx.createOutputBuffers()
    internal val deconlyIn = deconly.createInputBuffers()
    internal val deconlyOut = deconly.createOutputBuffers()
    internal val deconlyWIn = deconlyW?.createInputBuffers()
    internal val deconlyWOut = deconlyW?.createOutputBuffers()

    /** e.g. "lm:CPU dectx:NPU dec:GPU" — for diagnostics/UI. */
    val placements: String = if (lmSteps > 1) "${placement.label} ms$lmSteps" else placement.label

    // ---- host assets ------------------------------------------------------
    private val embChannel = RandomAccessFile(models.store.file(PocketTts.EMBED), "r").channel
    internal val embMap: ByteBuffer = embChannel
        .map(FileChannel.MapMode.READ_ONLY, 0, embChannel.size())
        .order(ByteOrder.LITTLE_ENDIAN)
    internal val inputLinear = readF32(models.store.file(PocketTts.INPUT_LINEAR))
    internal val bosInput = readF32(models.store.file(PocketTts.BOS))
    internal val neutral = readF32(models.store.file(PocketTts.NEUTRAL))
    val tokenizer = SpTokenizer(models.store.file(PocketTts.TOKENIZER))

    internal val endTokens: Set<Int> = tokenizer.encode(".!...?").drop(1).toSet()
    internal val fallbackTokens: Set<Int> = tokenizer.encode(",;:").drop(1).toSet()

    // ---- shared scratch ---------------------------------------------------
    // Every entry point runs under [lock] (or on the single worker), so one set
    // of buffers serves all sessions. Allocating per session cost ~25 MB and a
    // GC per utterance; the decoder arrays were re-allocated per text chunk.
    internal val pk = FloatArray(PocketTts.G * PocketTts.PMAX * PocketTts.HD)
    internal val pv = FloatArray(PocketTts.G * PocketTts.PMAX * PocketTts.HD)
    internal val mask = FloatArray(PocketTts.NH * (PocketTts.PMAX + 1))
    // Batched prompt prefill scratch (see PocketTtsSession.prefill).
    internal val prefillEmb = FloatArray(PocketTts.PREFILL_TOKENS * PocketTts.H)
    internal val prefillCos = FloatArray(PocketTts.PREFILL_TOKENS * PocketTts.HD)
    internal val prefillSin = FloatArray(PocketTts.PREFILL_TOKENS * PocketTts.HD)
    internal val prefillMask = FloatArray(PocketTts.PREFILL_TOKENS * (PocketTts.PMAX + 1))
    internal val prefillWrite = FloatArray(PocketTts.PREFILL_TOKENS * PocketTts.PMAX)
    internal val decFeat = FloatArray(PocketTts.MIMI_D * PocketTts.S_DEC)
    internal val decBlk = FloatArray((1 + PocketTts.F_BLK) * PocketTts.LDIM)
    internal val streamWin = FloatArray(PocketTts.MIMI_D * streamW)

    // ---- voice cache ------------------------------------------------------
    /** A repacked voice state, shared by every session that speaks it. */
    internal class VoiceState(val k: FloatArray, val v: FloatArray, val len: Int)

    /** LRU: switching voices should not re-read the ~3 MB file every request. */
    private val voiceCache = object : LinkedHashMap<String, VoiceState>(VOICE_CACHE, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, VoiceState>?): Boolean =
            size > VOICE_CACHE
    }

    internal fun voiceState(name: String): VoiceState = synchronized(voiceCache) {
        voiceCache.getOrPut(name) {
            val file = VoiceCatalog.fileFor(models, name)
                ?: throw java.io.FileNotFoundException(
                    "Missing voice cache '${PocketTts.voiceFile(name)}' under " +
                        "${PocketTts.VOICES_DIR}/ or the model root. Install one with " +
                        "scripts/download_voices.py or scripts/create_voice.py.",
                )
            val bb = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val t = bb.int
            check(t <= PocketTts.PMAX) { "voice state longer than KV capacity: $t > ${PocketTts.PMAX}" }
            val n = PocketTts.G * t * PocketTts.HD
            VoiceState(
                FloatArray(n) { android.util.Half.toFloat(bb.short) },
                FloatArray(n) { android.util.Half.toFloat(bb.short) },
                t,
            ).also {
                android.util.Log.i(
                    "PocketTTSTime",
                    "voice loaded: $name (${it.len} frames, cache=${voiceCache.size}/$VOICE_CACHE)",
                )
            }
        }
    }

    // ---- codec continuity -------------------------------------------------
    /**
     * The tail of the previous utterance: its last [PocketTts.F_BLK] latents and
     * the last [PocketTts.STREAM_L] feature positions it emitted. Priming the
     * next sentence with them keeps the dec_tx chain and the SEANet window warm,
     * which is what removes the cold-block onset transient at a sentence start.
     *
     * Engine-scoped rather than session-scoped because the TTS service issues
     * one request per sentence with a fresh session each. Guarded by the voice,
     * since splicing two different voices' tails would be wrong.
     */
    internal class CodecTail(val voice: String, val lats: List<FloatArray>, val feat: FloatArray)

    private var codecTail: CodecTail? = null

    internal fun codecTailFor(voice: String): CodecTail? =
        if (codecContinuity) codecTail?.takeIf { it.voice == voice } else null

    internal fun rememberCodecTail(tail: CodecTail?) {
        if (codecContinuity) codecTail = tail
    }

    // ---- concurrency ------------------------------------------------------
    internal val lock = Any()
    private val executor: ExecutorService =
        Executors.newSingleThreadExecutor { r -> Thread(r, "pockettts-synth").apply { isDaemon = true } }
    internal fun <T> submit(block: () -> T): Future<T> =
        executor.submit(java.util.concurrent.Callable { block() })

    init {
        android.util.Log.i(
            "PocketTTS",
            "engine ${placement.label} @ ${Placement.renderer()} (${lmGraphName}) " +
                "lmSig=${if (lmStepIn != null) 1 else 0} " +
                "prefill=${if (prefillIn != null) "${PocketTts.PREFILL_SIGNATURE}/${prefillIn.size}" else "none"} " +
                "heap=${Runtime.getRuntime().totalMemory() shr 20}MiB " +
                "native=${android.os.Debug.getNativeHeapAllocatedSize() shr 20}MiB",
        )
    }

    /** A fresh utterance. Sessions are cheap; the graphs stay here. */
    fun newSession(voice: String = voices.first().name): PocketTtsSession =
        PocketTtsSession(this, voice)

    /** One-shot convenience: synthesize [text] fully, blocking. */
    fun synthesize(text: String, voice: String = voices.first().name): TtsResult =
        newSession(voice).use { it.synthesize(text) }

    /** Streaming convenience: [onChunk] fires per decoded chunk, blocking. */
    fun stream(
        text: String,
        voice: String = voices.first().name,
        onChunk: (FloatArray) -> Unit,
    ): TtsResult = newSession(voice).use { it.stream(text, onChunk) }

    /** LM-only micro-benchmark (no decode); see [PocketTtsSession.microBenchLm]. */
    fun microBenchLm(steps: Int, voice: String = voices.first().name): TtsProfile =
        newSession(voice).use { it.microBenchLm(steps) }

    /** Download any required file the configured release source can provide. */
    fun ensureModels(onProgress: (Long, Long) -> Unit = { _, _ -> }) =
        models.ensure(requiredFiles(config), onProgress)

    override fun close() {
        executor.shutdownNow()
        gpuOpenClRunner?.close()
        listOf(
            lmIn, lmOut, lmResidentIn, lmResidentOut, lmMsIn, lmMsOut, dectxIn, dectxOut, deconlyIn, deconlyOut,
            deconlyWIn, deconlyWOut, prefillIn, prefillOut,
        ).forEach { l -> l?.forEach { it.close() } }
        lm.close(); lmMs?.close(); dectx.close(); deconly.close(); deconlyW?.close()
        embChannel.close()
        npuEnvironment?.close()
    }

    private fun readF32(f: File): FloatArray {
        val b = f.readBytes()
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.float }
    }

    companion object {
        /** Voices kept repacked in memory (each ~3 MB as fp32). */
        private const val VOICE_CACHE = 3

        /** Every model file a config needs, for `ensure()` and packaging. */
        fun requiredFiles(config: PocketTtsConfig): List<String> {
            val f = LinkedHashSet<String>()
            val lm = config.lmGraph ?: PocketTts.LM
            f += if (config.placement.lm == Accel.NPU) PocketTts.g5Variant(lm) else lm
            if (config.lmSteps > 1) {
                val ms = PocketTts.msGraph(config.lmSteps)
                f += if (config.placement.lm == Accel.NPU) PocketTts.g5Variant(ms) else ms
            }
            if (config.placement.dectx == Accel.NPU) {
                f += PocketTts.g5Variant(PocketTts.DEC_TX)
            }
            f += PocketTts.DEC_TX
            f += PocketTts.DECONLY
            f += PocketTts.deconlyGraph(config.streamW)
            f += listOf(
                PocketTts.EMBED, PocketTts.INPUT_LINEAR, PocketTts.BOS,
                PocketTts.NEUTRAL, PocketTts.TOKENIZER,
            )
            f += config.voices.map { PocketTts.voiceFile(it.name) }
            return f.toList()
        }
    }
}
