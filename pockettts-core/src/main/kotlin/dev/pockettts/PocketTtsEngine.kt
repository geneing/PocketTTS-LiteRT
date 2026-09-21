package dev.pockettts

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future

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
    val noiseSeed: Long? = config.noiseSeed
    private val gpuCache: File? = config.gpuCache

    /** Per-graph compile time (ms), keyed lm/lm_ms/dectx/dec/dec_w. */
    val loadMs = LinkedHashMap<String, Long>()

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
            CompiledModel.create(p, CompiledModel.Options(Accelerator.CPU), null)
        }
        loadMs[key] = (System.nanoTime() - t) / 1_000_000
        return model
    }

    // ---- graphs -----------------------------------------------------------
    val lmGraphName: String = config.lmGraph
        ?: if (models.store.exists(PocketTts.LM_INT8)) PocketTts.LM_INT8 else PocketTts.LM

    internal val lm: CompiledModel = load(lmGraphName, "lm", placement.lm)
    internal val lmMs: CompiledModel? =
        if (lmSteps > 1) load(PocketTts.msGraph(lmSteps), "lm_ms", placement.lm) else null
    internal val dectx: CompiledModel = load(PocketTts.DEC_TX, "dectx", placement.dectx)
    internal val deconly: CompiledModel = load(PocketTts.DECONLY, "dec", placement.deconly)
    internal val deconlyW: CompiledModel? = if (models.store.exists(PocketTts.deconlyGraph(streamW))) {
        load(PocketTts.deconlyGraph(streamW), "dec_w", placement.deconly)
    } else {
        null
    }

    internal val lmIn = lm.createInputBuffers()
    internal val lmOut = lm.createOutputBuffers()
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

    // ---- concurrency ------------------------------------------------------
    internal val lock = Any()
    private val executor: ExecutorService =
        Executors.newSingleThreadExecutor { r -> Thread(r, "pockettts-synth").apply { isDaemon = true } }
    internal fun <T> submit(block: () -> T): Future<T> =
        executor.submit(java.util.concurrent.Callable { block() })

    init {
        android.util.Log.i(
            "PocketTTS",
            "engine ${placement.label} @ ${Placement.renderer()} (${lmGraphName})",
        )
    }

    /** A fresh utterance. Sessions are cheap; the graphs stay here. */
    fun newSession(voice: String = PocketTts.VOICES.first()): PocketTtsSession =
        PocketTtsSession(this, voice)

    /** One-shot convenience: synthesize [text] fully, blocking. */
    fun synthesize(text: String, voice: String = PocketTts.VOICES.first()): TtsResult =
        newSession(voice).use { it.synthesize(text) }

    /** Streaming convenience: [onChunk] fires per decoded chunk, blocking. */
    fun stream(
        text: String,
        voice: String = PocketTts.VOICES.first(),
        onChunk: (FloatArray) -> Unit,
    ): TtsResult = newSession(voice).use { it.stream(text, onChunk) }

    /** LM-only micro-benchmark (no decode); see [PocketTtsSession.microBenchLm]. */
    fun microBenchLm(steps: Int, voice: String = PocketTts.VOICES.first()): TtsProfile =
        newSession(voice).use { it.microBenchLm(steps) }

    /** Download any required file the configured release source can provide. */
    fun ensureModels(onProgress: (Long, Long) -> Unit = { _, _ -> }) =
        models.ensure(requiredFiles(config), onProgress)

    override fun close() {
        executor.shutdownNow()
        listOf(lmIn, lmOut, lmMsIn, lmMsOut, dectxIn, dectxOut, deconlyIn, deconlyOut, deconlyWIn, deconlyWOut)
            .forEach { l -> l?.forEach { it.close() } }
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
        /** Every model file a config needs, for `ensure()` and packaging. */
        fun requiredFiles(config: PocketTtsConfig): List<String> {
            val f = LinkedHashSet<String>()
            f += config.lmGraph
                ?: if (config.models.store.exists(PocketTts.LM_INT8)) PocketTts.LM_INT8 else PocketTts.LM
            if (config.lmSteps > 1) f += PocketTts.msGraph(config.lmSteps)
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
            PocketTts.VOICES.forEach { f += PocketTts.voiceFile(it) }
            return f.toList()
        }
    }
}
