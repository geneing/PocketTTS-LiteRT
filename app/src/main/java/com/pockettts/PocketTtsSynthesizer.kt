package com.pockettts

import android.content.Context
import android.util.Half
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Random
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pocket TTS (Kyutai, 100M) on LiteRT CompiledModel.
 *
 * Pocket TTS is a flow-matching LM over continuous 32-dim Mimi latents: per
 * 12.5 Hz frame a 6-layer/1024-wide causal transformer conditions a 6-block
 * AdaLN MLP flow head that turns one Gaussian draw into the next latent
 * (Lagrangian Self Distillation, 1 step — no iterative sampling loop); a 20M
 * tiny Mimi (x16 ConvTranspose upsample + 2-layer transformer + SEANet)
 * decodes latents to 24 kHz audio. The voice is a precomputed prompt KV cache
 * (`pt_voice_*.bin`, repacked from Kyutai's published per-voice states).
 *
 * Four graphs, all stateless with host-side state (the dia2/vibevoice
 * packed-KV pattern):
 *  * `pt_flowlm_step`  — one AR step; packed KV `[1,96,512,64]` in/out.
 *  * `pt_flow_head`    — cond + noise -> latent (LSD time embeds baked in).
 *  * `pt_mimi_dec_tx`  — 64-latent-frame block of the Mimi decoder
 *    transformer; blocks overlap 32 frames because the 2-layer sliding-window
 *    (250) attention has a stacked receptive field of 498 positions.
 *  * `pt_mimi_deconly` — SEANet decoder, one-shot 256-frame window (causal,
 *    so real frames are exact regardless of the zero tail).
 *
 * Text chunking, EOS handling and the noise schedule mirror the reference
 * pocket_tts Python package; the sentencepiece unigram tokenizer is ported in
 * [SpTokenizer]. Host-vs-reference parity of every graph and of the full
 * pipeline is checked in scripts/build_pockettts.py.
 */
class PocketTtsSynthesizer(
    context: Context,
    /** Per-graph accelerator choice; defaults to the device policy + force_* files. */
    val placement: Placement = Placement.default(context),
    /** When set, every [synthesize] reseeds the noise RNG so repeats are identical. */
    private val noiseSeed: Long? = null,
    /** When set, GPU graphs serialize their compiled program cache here (load-time studies). */
    private val gpuCache: File? = null,
    /**
     * Frames per LM invocation. 1 = the shipped single-step fused graph. >1 loads
     * `pt_flowlm_ms{N}_fp16.tflite` in addition to the 1-step graph (and uses it
     * for the prompt/tail), which amortizes the per-invocation upload + readback
     * sync over N frames. See docs/multistep_lm_plan.md.
     */
    private val lmSteps: Int = 1,
    /**
     * Override the flow-LM graph filename, e.g. `pt_flowlm_fused_dyn8_all.tflite`.
     * null = the shipped [LM]. Used by the benchmark to compare quantizations of
     * the same graph without rebuilding the app.
     */
    private val lmGraph: String? = null,
    /** SEANet window (feature positions) for [synthesizeStream]; see [STREAM_W]. */
    private val streamW: Int = STREAM_W,
) : Closeable {

    companion object {
        const val H = 1024               // flow-LM width
        const val HD = 64                // head dim
        const val NH = 16                // heads
        const val LAYERS = 6
        const val G = LAYERS * NH        // packed KV groups
        const val PMAX = 512             // KV capacity: voice + text + audio frames
        const val LDIM = 32              // Mimi latent dim
        const val THETA = 10000.0

        const val UPS = 16               // 12.5 Hz -> 200 Hz
        const val MIMI_D = 512
        const val F_BLK = 64             // dec_tx block payload frames
        const val F_HOP = 32             // dec_tx block hop
        const val S_BLK = F_BLK * UPS
        const val DEC_FRAMES = 256       // deconly window frames
        const val S_DEC = DEC_FRAMES * UPS
        const val SPF = 1920             // samples per 12.5 Hz frame
        /** Samples per Mimi feature position: SPF / UPS = 24000 / 200 Hz. */
        const val SPP = SPF / UPS
        const val SAMPLE_RATE = 24000

        // Generation defaults from the english config / pocket_tts defaults.
        const val TEMP = 0.3f
        const val EOS_THRESHOLD = -4.0f
        const val MAX_TOKENS_PER_CHUNK = 50
        const val TOKENS_PER_SECOND = 3.0
        const val GEN_SECONDS_PADDING = 2.0
        const val FRAME_RATE = 12.5
        const val MASK_NEG = -1e4f

        // step + flow head fused into one graph with one output tensor: on
        // Mali the per-frame cost is dispatch/sync-bound, and two invocations
        // plus four readbacks per frame cost more than the math itself.
        const val LM = "pt_flowlm_fused_fp16.tflite"

        /**
         * Dynamic-range int8 flow-LM: int8 weights, fp32 activations, so the
         * inputs and output stay fp32 and the host protocol is unchanged.
         * Measured 2.27x faster per frame than [LM] on XNNPACK (19.6 -> 8.65 ms),
         * 2.16x vs 1.51x end to end, and 82 MB instead of 161 MB; it passes the
         * by-ear test against the fp16 graph. See docs/RESULTS.md.
         */
        const val LM_INT8 = "pt_flowlm_fused_dyn8_all.tflite"

        /**
         * Prefer the int8 flow-LM when it has been pushed, else the fp16 one.
         * The fp16 graph is not a fallback in the error sense -- both are valid
         * builds; this only decides which one a fresh device uses.
         */
        fun lmGraphFor(dir: File) = if (File(dir, LM_INT8).exists()) LM_INT8 else LM

        /** N-step fused decode graph: N frames per invocation (see docs/multistep_lm_plan.md). */
        fun msGraph(n: Int) = "pt_flowlm_ms${n}_fp16.tflite"
        const val DEC_TX = "pt_mimi_dec_tx_fp16.tflite"
        const val DECONLY = "pt_mimi_deconly_fp16.tflite"

        /**
         * SEANet window in feature positions for streaming. 512 (32 latent
         * frames) measures fastest on both RTF and time-to-first-audio on the
         * Pixel 10 GPU: it is the smallest window that still leaves a useful
         * margin over the 16-position overlap, and it emits 3 chunks instead of
         * 1-2. The host (XNNPACK) is bit-exact at every window size; on the GPU
         * the delegate picks a different convolution algorithm above ~1024, so
         * the small windows agree with each other but differ from the 2048/4096
         * graphs by ~-73 dBFS rms. See docs/streaming.md.
         */
        const val STREAM_W = 512

        /**
         * Left context kept when sliding the SEANet window. The decoder's
         * measured left receptive field is ~7.6 feature positions; 16 is a
         * safety margin and still leaves 15 frames of new audio per window at
         * [STREAM_W] = 512. Confirmed exact on the host for every L >= 8.
         */
        const val STREAM_L = 16

        /** Smaller-window SEANet decoder, from `build_pockettts.py stream`. */
        fun deconlyGraph(w: Int) = "pt_mimi_deconly_w${w}_fp16.tflite"
        const val EMBED = "pt_embed_f16.bin"
        const val INPUT_LINEAR = "pt_input_linear_f32.bin"
        const val BOS = "pt_bos_input_f32.bin"
        const val NEUTRAL = "pt_neutral_latent_f32.bin"
        const val TOKENIZER = "pt_tokenizer.tsv"

        // CC-BY-4.0 (alba-mackenna, VCTK) and CC0 (voice-donations, voice-zero)
        // voices only; the CC-BY-NC ones (expresso, ears) are not bundled.
        val VOICES = listOf("alba", "marius", "javert", "charles", "mary", "eve")
    }

    private val modelDir =
        requireNotNull(context.getExternalFilesDir(null)) { "External storage unavailable" }

    private val appCtx = context.applicationContext

    /**
     * AOT-compiled Tensor G5 model name for a stock graph, e.g.
     * `pt_flowlm_fused_fp16.tflite` -> `pt_flowlm_fused_fp16_g5.tflite`. The
     * compiled artifact is a *different file*: it holds a Google Tensor
     * dispatch partition in place of the original ops, so it cannot be produced
     * by asking the stock graph for the NPU accelerator.
     */
    private fun g5Variant(name: String) = name.replace(".tflite", "_g5.tflite")

    private var npuEnvironment: Environment? = null

    /**
     * The LiteRT environment the NPU needs. The Google Tensor dispatch shim is
     * `dlopen`ed by absolute path from the app's native library directory, so
     * that directory has to be handed over explicitly -- an unset value only
     * logs a warning and the model quietly never reaches the NPU.
     */
    private fun npuEnv(): Environment {
        npuEnvironment?.let { return it }
        val env = Environment.create(
            appCtx,
            mapOf(
                Environment.Option.DispatchLibraryDir to
                    appCtx.applicationInfo.nativeLibraryDir,
            ),
        )
        npuEnvironment = env
        return env
    }

    private fun path(name: String): File {
        val f = File(modelDir, name)
        check(f.exists()) { "Missing $name — push files first: scripts/install_to_device.sh" }
        return f
    }

    // Per-graph compile time (ms), keyed lm/dectx/dec. Populated as graphs load.
    val loadMs = LinkedHashMap<String, Long>()

    /**
     * Compile one graph on its accelerator. GPU loads fall back to CPU on
     * failure (fp16 weights dequantize to fp32 there); load time is recorded
     * either way. `gpuCache`, when set, enables the delegate's serialized
     * program cache so a reload skips kernel compilation.
     *
     * NPU loads the `*_g5.tflite` AOT-compiled variant and does *not* fall back:
     * the compiled partition only has a dispatch kernel, so a CPU retry of the
     * same file cannot work, and a silent fallback is exactly how a mis-wired
     * NPU ends up reporting a CPU-speed number.
     */
    private fun load(name: String, key: String, accel: Accel): CompiledModel {
        val file = if (accel == Accel.NPU) g5Variant(name) else name
        val p = path(file).absolutePath
        val t = System.nanoTime()
        val model = try {
            when (accel) {
                Accel.CPU -> CompiledModel.create(p, CompiledModel.Options(Accelerator.CPU), null)
                Accel.GPU, Accel.GPU32 -> {
                    val opts = CompiledModel.Options(Accelerator.GPU)
                    val gpu = if (accel == Accel.GPU32) {
                        CompiledModel.GpuOptions(
                            precision = CompiledModel.GpuOptions.Precision.FP32,
                        )
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
                Accel.NPU -> CompiledModel.create(
                    p,
                    CompiledModel.Options(Accelerator.NPU),
                    npuEnv(),
                )
            }
        } catch (e: Throwable) {
            if (accel == Accel.NPU) throw e
            CompiledModel.create(p, CompiledModel.Options(Accelerator.CPU), null)
        }
        loadMs[key] = (System.nanoTime() - t) / 1_000_000
        return model
    }

    /** Filename of the flow-LM graph actually loaded (see [lmGraph]). */
    val lmGraphName: String = lmGraph ?: lmGraphFor(modelDir)

    val lm = load(lmGraphName, "lm", placement.lm)

    /** N-step decode graph; null when [lmSteps] == 1. Prompt/tail still use [lm]. */
    private val lmMs: CompiledModel? =
        if (lmSteps > 1) load(msGraph(lmSteps), "lm_ms", placement.lm) else null

    val dectx = load(DEC_TX, "dectx", placement.dectx)
    val deconly = load(DECONLY, "dec", placement.deconly)

    /**
     * Streaming SEANet window. null when the smaller graph is not installed, in
     * which case [synthesizeStream] falls back to the one-shot decode path.
     */
    val deconlyW: CompiledModel? =
        if (File(modelDir, deconlyGraph(streamW)).exists()) {
            load(deconlyGraph(streamW), "dec_w", placement.deconly)
        } else {
            null
        }
    private val deconlyWIn = deconlyW?.createInputBuffers()
    private val deconlyWOut = deconlyW?.createOutputBuffers()

    /** e.g. "lm:GPU dectx:CPU dec:GPU" — shown in the UI status line. */
    val placements = if (lmSteps > 1) "${placement.label} ms$lmSteps" else placement.label

    private val lmIn = lm.createInputBuffers()
    private val lmOut = lm.createOutputBuffers()
    private val lmMsIn = lmMs?.createInputBuffers()
    private val lmMsOut = lmMs?.createOutputBuffers()
    private val dectxIn = dectx.createInputBuffers()
    private val dectxOut = dectx.createOutputBuffers()
    private val deconlyIn = deconly.createInputBuffers()
    private val deconlyOut = deconly.createOutputBuffers()

    // ---- host assets ------------------------------------------------------
    private val embChannel = RandomAccessFile(path(EMBED), "r").channel
    private val embMap = embChannel
        .map(FileChannel.MapMode.READ_ONLY, 0, embChannel.size()).order(ByteOrder.LITTLE_ENDIAN)
    private val inputLinear = readF32(path(INPUT_LINEAR))      // [1024, 32] row-major
    private val bosInput = readF32(path(BOS))                  // [1024]
    private val neutral = readF32(path(NEUTRAL))               // [32]
    val tokenizer = SpTokenizer(path(TOKENIZER))

    private val endTokens: Set<Int>
    private val fallbackTokens: Set<Int>

    init {
        endTokens = tokenizer.encode(".!...?").drop(1).toSet()
        fallbackTokens = tokenizer.encode(",;:").drop(1).toSet()
        android.util.Log.i("PocketTTS", "placement ${placement.label} @ ${Placement.renderer()}")
    }

    // ---- host state -------------------------------------------------------
    private val pk = FloatArray(G * PMAX * HD)
    private val pv = FloatArray(G * PMAX * HD)
    private val mask = FloatArray(NH * (PMAX + 1))
    private var pos = 0
    // Per-utterance stage accumulators (ns / counts), reset by each synthesize.
    private var sLmIn = 0L
    private var sLmRun = 0L
    private var sLmRead = 0L
    private var sLmSteps = 0
    private var sLmInv = 0
    private var sLmInBytes = 0L
    private var sLmOutBytes = 0L
    private var sPrompt = 0
    private var sFrames = 0
    private var sDecTx = 0L
    private var sSeanet = 0L
    private var sChunks = 0
    private var sFirstChunk = -1L
    private var sAudioChunks = 0

    private var voiceName = ""
    private var voiceK = FloatArray(0)
    private var voiceV = FloatArray(0)
    private var voiceLen = 0

    private val cosArr = FloatArray(HD)
    private val sinArr = FloatArray(HD)
    private val invFreq = DoubleArray(HD / 2) { 1.0 / Math.pow(THETA, it / 32.0) }
    private val rnd = Random(noiseSeed ?: System.nanoTime())

    /** Stage timings for one [synthesize] call (ms unless noted). */
    data class Profile(
        val lmSteps: Int,
        val promptSteps: Int,
        val genFrames: Int,
        val lmInMs: Long,
        val lmRunMs: Long,
        val lmReadMs: Long,
        val decTxMs: Long,
        val seanetMs: Long,
        val chunks: Int,
        /** LM graph invocations: == [lmSteps] for the 1-step graph, fewer for an N-step graph. */
        val lmInvocations: Int = 0,
        /** Bytes written to the LM input buffers this call (packed KV dominates). */
        val lmInBytes: Long = 0,
        /** Bytes read from the LM output buffer this call. */
        val lmOutBytes: Long = 0,
        /**
         * Time from the start of [synthesizeStream] to the first audio chunk,
         * ms. -1 for the one-shot path, which has no chunks.
         */
        val firstChunkMs: Long = -1,
        /** Audio chunks emitted by the streaming decoder. */
        val audioChunks: Int = 0,
    )

    data class Result(val audio: FloatArray, val frames: Int, val ms: Long, val profile: Profile)

    /** Load a repacked voice state: int32 T, then k and v as fp16 `[96][T][64]`. */
    fun loadVoice(name: String) {
        if (name == voiceName) return
        val bb = ByteBuffer.wrap(path("pt_voice_$name.bin").readBytes())
            .order(ByteOrder.LITTLE_ENDIAN)
        val t = bb.int
        check(t <= PMAX) { "voice state longer than KV capacity: $t > $PMAX" }
        val n = G * t * HD
        val k = FloatArray(n) { Half.toFloat(bb.short) }
        val v = FloatArray(n) { Half.toFloat(bb.short) }
        voiceK = k; voiceV = v; voiceLen = t; voiceName = name
    }

    private fun resetToVoice() {
        pk.fill(0f); pv.fill(0f)
        for (g in 0 until G) {
            System.arraycopy(voiceK, g * voiceLen * HD, pk, g * PMAX * HD, voiceLen * HD)
            System.arraycopy(voiceV, g * voiceLen * HD, pv, g * PMAX * HD, voiceLen * HD)
        }
        mask.fill(MASK_NEG)
        for (h in 0 until NH) {
            val base = h * (PMAX + 1)
            for (p in 0 until voiceLen) mask[base + p] = 0f
            mask[base + PMAX] = 0f                        // current token, concatenated at tail
        }
        pos = voiceLen
    }

    // ---- small host math --------------------------------------------------
    private fun embRow(id: Int): FloatArray {
        val out = FloatArray(H)
        var b = id * H * 2
        for (j in 0 until H) { out[j] = Half.toFloat(embMap.getShort(b)); b += 2 }
        return out
    }

    private fun projectLatent(lat: FloatArray): FloatArray {
        val out = FloatArray(H)
        for (o in 0 until H) {
            var acc = 0f
            val row = o * LDIM
            for (i in 0 until LDIM) acc += inputLinear[row + i] * lat[i]
            out[o] = acc
        }
        return out
    }

    private fun ropeFill(p: Int) {
        for (j in 0 until HD / 2) {
            val ang = p * invFreq[j]
            val c = cos(ang).toFloat(); val s = sin(ang).toFloat()
            cosArr[j] = c; cosArr[j + HD / 2] = c
            sinArr[j] = s; sinArr[j + HD / 2] = s
        }
    }

    private val zeroNoise = FloatArray(LDIM)

    private fun gaussNoise(): FloatArray = FloatArray(LDIM) {
        (rnd.nextGaussian() * sqrt(TEMP.toDouble())).toFloat()
    }

    // Multi-step decode graph scratch (sized for [lmSteps] frames).
    private val msCos = FloatArray(lmSteps * HD)
    private val msSin = FloatArray(lmSteps * HD)
    private val msMask = FloatArray(lmSteps * (PMAX + 1))
    private val msWrite = FloatArray(lmSteps * PMAX)
    private val msNoise = FloatArray(lmSteps * LDIM)

    /**
     * One fused frame: flow-LM step + flow head in a single invocation.
     * Output layout: eos(1) | latent(32) | new-k(96*64) | new-v(96*64).
     * Returns (latent, eosLogit) and appends this step's K/V at [pos].
     * Text prompting passes zero noise and ignores the latent.
     */
    private fun step(emb: FloatArray, noise: FloatArray): Pair<FloatArray, Float> {
        check(pos < PMAX) { "KV cache overflow at $pos" }
        val t0 = System.nanoTime()
        ropeFill(pos)
        lmIn[0].writeFloat(emb)
        lmIn[1].writeFloat(cosArr)
        lmIn[2].writeFloat(sinArr)
        lmIn[3].writeFloat(mask)
        lmIn[4].writeFloat(pk)
        lmIn[5].writeFloat(pv)
        lmIn[6].writeFloat(noise)
        val t1 = System.nanoTime()
        lm.run(lmIn, lmOut)
        val t2 = System.nanoTime()
        val out = lmOut[0].readFloat()
        val eos = out[0]
        val latent = out.copyOfRange(1, 1 + LDIM)
        val kvBase = 1 + LDIM
        for (g in 0 until G) {
            System.arraycopy(out, kvBase + g * HD, pk, g * PMAX * HD + pos * HD, HD)
            System.arraycopy(out, kvBase + G * HD + g * HD, pv, g * PMAX * HD + pos * HD, HD)
        }
        for (h in 0 until NH) mask[h * (PMAX + 1) + pos] = 0f
        pos++
        val t3 = System.nanoTime()
        sLmIn += t1 - t0; sLmRun += t2 - t1; sLmRead += t3 - t2; sLmSteps++
        sLmInv++
        sLmInBytes += (emb.size + cosArr.size + sinArr.size + mask.size +
            pk.size + pv.size + noise.size).toLong() * Float.SIZE_BYTES
        sLmOutBytes += out.size.toLong() * Float.SIZE_BYTES
        return latent to eos
    }

    /**
     * [lmSteps] frames in one invocation of the multi-step graph. Input `emb` is
     * frame 0's embedding; frames 1..N-1 are fed in-graph from the projected
     * latent. The graph uploads the packed KV once, appends all N new rows
     * internally with the one-hot `msWrite` mask, and returns only the new rows;
     * this splices them into the host mirror so the 25 MB cache is never read back.
     * Output layout: eos[N] | latent[N,32] | new-k[N,G,64] | new-v[N,G,64].
     */
    private fun stepMulti(emb: FloatArray, noises: Array<FloatArray>): Pair<Array<FloatArray>, FloatArray> {
        val n = lmSteps
        check(pos + n <= PMAX) { "multi-step KV cache overflow at $pos + $n" }
        val t0 = System.nanoTime()
        java.util.Arrays.fill(msWrite, 0f)
        for (i in 0 until n) {
            ropeFill(pos + i)
            System.arraycopy(cosArr, 0, msCos, i * HD, HD)
            System.arraycopy(sinArr, 0, msSin, i * HD, HD)
            val mb = i * (PMAX + 1)
            for (p in 0..PMAX) msMask[mb + p] = if (p < pos + i || p == PMAX) 0f else MASK_NEG
            msWrite[i * PMAX + (pos + i)] = 1f
            System.arraycopy(noises[i], 0, msNoise, i * LDIM, LDIM)
        }
        val ins = requireNotNull(lmMsIn) { "multi-step graph not loaded" }
        val outs = requireNotNull(lmMsOut)
        val model = requireNotNull(lmMs)
        ins[0].writeFloat(emb)
        ins[1].writeFloat(msCos)
        ins[2].writeFloat(msSin)
        ins[3].writeFloat(msMask)
        ins[4].writeFloat(msWrite)
        ins[5].writeFloat(pk)
        ins[6].writeFloat(pv)
        ins[7].writeFloat(msNoise)
        val t1 = System.nanoTime()
        model.run(ins, outs)
        val t2 = System.nanoTime()
        val out = outs[0].readFloat()
        val eos = FloatArray(n)
        val lats = Array(n) { FloatArray(LDIM) }
        for (i in 0 until n) {
            eos[i] = out[i]
            System.arraycopy(out, n + i * LDIM, lats[i], 0, LDIM)
        }
        var o = n * (1 + LDIM)
        for (i in 0 until n) {
            val p = pos + i
            for (g in 0 until G) {
                System.arraycopy(out, o, pk, g * PMAX * HD + p * HD, HD)
                o += HD
            }
        }
        for (i in 0 until n) {
            val p = pos + i
            for (g in 0 until G) {
                System.arraycopy(out, o, pv, g * PMAX * HD + p * HD, HD)
                o += HD
            }
        }
        for (h in 0 until NH) {
            val base = h * (PMAX + 1)
            for (i in 0 until n) mask[base + pos + i] = 0f
        }
        pos += n
        val t3 = System.nanoTime()
        sLmIn += t1 - t0; sLmRun += t2 - t1; sLmRead += t3 - t2; sLmSteps += n; sLmInv++
        sLmInBytes += (H + 2 * n * HD + n * (PMAX + 1) + n * PMAX +
            2 * G * PMAX * HD + n * LDIM).toLong() * Float.SIZE_BYTES
        sLmOutBytes += out.size.toLong() * Float.SIZE_BYTES
        return lats to eos
    }

    /** Generate speech for `text` with the currently loaded voice. */
    fun synthesize(text: String, voice: String): Result {
        val t0 = System.nanoTime()
        noiseSeed?.let { rnd.setSeed(it) }
        resetProfile()
        loadVoice(voice)
        val audio = ArrayList<FloatArray>()
        var frames = 0
        val chunks = splitIntoBestSentences(text)
        for (chunk in chunks) {
            val (prepared, eosGuess) = prepareTextPrompt(chunk)
            val ids = tokenizer.encode(prepared)
            val latents = generateChunk(ids, framesAfterEos = eosGuess + 2)
            android.util.Log.i(
                "PocketTTS",
                "chunk: ${ids.size} tokens -> ${latents.size} frames",
            )
            frames += latents.size
            sPrompt += ids.size; sFrames += latents.size; sChunks++
            if (latents.isNotEmpty()) audio.add(decode(latents))
        }
        val total = audio.sumOf { it.size }
        val out = FloatArray(total)
        var o = 0
        for (a in audio) { System.arraycopy(a, 0, out, o, a.size); o += a.size }
        return Result(out, frames, (System.nanoTime() - t0) / 1_000_000, snapshotProfile())
    }

    /** Zero the per-call stage accumulators. */
    private fun resetProfile() {
        sLmIn = 0; sLmRun = 0; sLmRead = 0; sLmSteps = 0; sLmInv = 0
        sLmInBytes = 0; sLmOutBytes = 0
        sPrompt = 0; sFrames = 0; sDecTx = 0; sSeanet = 0; sChunks = 0
        sFirstChunk = -1; sAudioChunks = 0
    }

    private fun snapshotProfile() = Profile(
        lmSteps = sLmSteps,
        promptSteps = sPrompt,
        genFrames = sFrames,
        lmInMs = sLmIn / 1_000_000,
        lmRunMs = sLmRun / 1_000_000,
        lmReadMs = sLmRead / 1_000_000,
        decTxMs = sDecTx / 1_000_000,
        seanetMs = sSeanet / 1_000_000,
        chunks = sChunks,
        lmInvocations = sLmInv,
        lmInBytes = sLmInBytes,
        lmOutBytes = sLmOutBytes,
        firstChunkMs = sFirstChunk,
        audioChunks = sAudioChunks,
    )

    /**
     * LM-only micro-benchmark: [steps] autoregressive frames from the voice
     * prompt with no Mimi decode, so the LM's per-invocation input/run/read
     * cost can be compared across placements without decoder noise. When
     * [noiseSeed] is set every repeat is identical.
     */
    fun microBenchLm(steps: Int, voice: String = VOICES.first()): Profile {
        noiseSeed?.let { rnd.setSeed(it) }
        resetProfile()
        loadVoice(voice)
        resetToVoice()
        val n = minOf(steps, PMAX - pos - 1).coerceAtLeast(0)
        var emb = bosInput
        var g = 0
        while (g < n) {
            if (lmSteps > 1 && g + lmSteps <= n && pos + lmSteps <= PMAX) {
                val (lats, _) = stepMulti(emb, Array(lmSteps) { gaussNoise() })
                emb = projectLatent(lats[lmSteps - 1])
                g += lmSteps
            } else {
                emb = projectLatent(step(emb, gaussNoise()).first)
                g++
            }
        }
        sFrames = n
        return snapshotProfile()
    }

    /**
     * The reference autoregressive loop for one <=50-token chunk. [sink], when
     * given, receives each latent as it is produced — that is what lets the
     * decoder run behind the LM instead of after it.
     */
    private fun generateChunk(
        ids: IntArray,
        framesAfterEos: Int,
        sink: ((FloatArray) -> Unit)? = null,
    ): List<FloatArray> {
        resetToVoice()
        for (id in ids) step(embRow(id), zeroNoise)
        val estimate = ceil((ids.size / TOKENS_PER_SECOND + GEN_SECONDS_PADDING) * FRAME_RATE)
        val maxGen = minOf(estimate.toInt(), PMAX - pos - 1)
        val latents = ArrayList<FloatArray>(maxGen)
        var emb = bosInput
        var eosStep = -1
        var g = 0
        while (g < maxGen) {
            if (lmSteps > 1 && g + lmSteps <= maxGen && pos + lmSteps <= PMAX) {
                val (lats, eoses) = stepMulti(emb, Array(lmSteps) { gaussNoise() })
                var stop = false
                for (i in 0 until lmSteps) {
                    if (eoses[i] > EOS_THRESHOLD && eosStep < 0) eosStep = g + i
                    if (eosStep >= 0 && g + i >= eosStep + framesAfterEos) { stop = true; break }
                    latents.add(lats[i]); sink?.invoke(lats[i])
                }
                if (stop) break
                emb = projectLatent(lats[lmSteps - 1])
                g += lmSteps
            } else {
                val (lat, eosLogit) = step(emb, gaussNoise())
                if (eosLogit > EOS_THRESHOLD && eosStep < 0) eosStep = g
                if (eosStep >= 0 && g >= eosStep + framesAfterEos) break
                latents.add(lat); sink?.invoke(lat)
                emb = projectLatent(lat)
                g++
            }
        }
        return latents
    }

    /** Mimi decode: overlapped dec_tx blocks -> one-shot SEANet window. */
    private fun decode(latents: List<FloatArray>): FloatArray {
        val tDec = System.nanoTime()
        val t = minOf(latents.size, DEC_FRAMES)
        val feat = FloatArray(MIMI_D * S_DEC)
        val blk = FloatArray((1 + F_BLK) * LDIM)

        fun runBlock(prev: FloatArray, start: Int): FloatArray {
            System.arraycopy(prev, 0, blk, 0, LDIM)
            for (f in 0 until F_BLK) {
                val src = if (start + f < t) latents[start + f] else neutral
                System.arraycopy(src, 0, blk, (1 + f) * LDIM, LDIM)
            }
            dectxIn[0].writeFloat(blk)
            dectx.run(dectxIn, dectxOut)
            return dectxOut[0].readFloat()               // [512 * 1024]
        }

        var out = runBlock(neutral, 0)
        val n0 = minOf(F_BLK, t)
        for (c in 0 until MIMI_D)
            System.arraycopy(out, c * S_BLK, feat, c * S_DEC, n0 * UPS)
        var kept = F_BLK
        while (kept < t) {
            val start = kept - F_HOP
            out = runBlock(latents[start - 1], start)
            val n = minOf(F_BLK, t - start)
            val keepN = (n - F_HOP) * UPS
            for (c in 0 until MIMI_D)
                System.arraycopy(out, c * S_BLK + F_HOP * UPS, feat, c * S_DEC + kept * UPS, keepN)
            kept += n - F_HOP
        }

        val tSeanet = System.nanoTime()
        deconlyIn[0].writeFloat(feat)
        deconly.run(deconlyIn, deconlyOut)
        val wav = deconlyOut[0].readFloat()
        sDecTx += tSeanet - tDec
        sSeanet += System.nanoTime() - tSeanet
        return FloatArray(t * SPF) { wav[it].coerceIn(-1f, 1f) }
    }

    // ---- streaming decode --------------------------------------------------

    /**
     * Incremental Mimi decode. Frames go in as the LM produces them; a dec_tx
     * block runs as soon as its 64 frames (or the chunk end) exist, and a
     * SEANet window runs as soon as [STREAM_W] feature positions are available.
     * Audio leaves through [onChunk].
     *
     * Both stages are block-exact rather than approximate: dec_tx keeps only the
     * region its 32-frame overlap makes valid, and the SEANet is strictly causal
     * (output sample s depends only on feature positions <= s/UPS) with a ~8
     * position left receptive field, so a sliding window reproduces the one-shot
     * 4096-position run. Concatenated chunks equal [decode]'s output bit-for-bit
     * on the host; on the GPU they differ by the delegate's own rounding
     * (~-73 dBFS rms) because it picks a different convolution algorithm for the
     * smaller window. See docs/streaming.md.
     */
    private inner class StreamDecoder(private val onChunk: (FloatArray) -> Unit) {
        private val lats = ArrayList<FloatArray>()
        private val feat = FloatArray(MIMI_D * S_DEC)
        private val win = FloatArray(MIMI_D * streamW)
        private val blk = FloatArray((1 + F_BLK) * LDIM)
        private var kept = 0        // frames already written into `feat`
        private var featPos = 0     // feature positions written
        private var emitted = 0     // feature positions already emitted as audio
        var chunks = 0
            private set

        fun push(lat: FloatArray) {
            lats.add(lat)
            advance(final = false)
            while (featPos - emitted >= streamW - STREAM_L) emitWindow()
        }

        fun flush() {
            advance(final = true)
            while (featPos > emitted) emitWindow()
        }

        /** Run every dec_tx block whose inputs are complete. */
        private fun advance(final: Boolean) {
            val n = lats.size
            if (n == 0) return
            if (kept == 0) {
                // Block 0 needs all F_BLK frames; only a flush may run it short.
                if (n < F_BLK && !final) return
                block(n, 0, neutral)
            }
            while (kept < n && (final || n - kept >= F_HOP)) {
                block(n, kept - F_HOP, lats[kept - F_HOP - 1])
            }
        }

        /**
         * One dec_tx block over frames [start, start+F_BLK); appends the newly
         * valid frames to `feat`. Block 0 keeps from frame 0, later blocks drop
         * the first F_HOP frames, which the previous block already kept.
         */
        private fun block(size: Int, start: Int, prev: FloatArray) {
            System.arraycopy(prev, 0, blk, 0, LDIM)
            for (f in 0 until F_BLK) {
                val src = if (start + f < size) lats[start + f] else neutral
                System.arraycopy(src, 0, blk, (1 + f) * LDIM, LDIM)
            }
            dectxIn[0].writeFloat(blk)
            val t0 = System.nanoTime()
            dectx.run(dectxIn, dectxOut)
            val out = dectxOut[0].readFloat()
            sDecTx += System.nanoTime() - t0
            val drop = if (start == 0) 0 else F_HOP
            val keepN = minOf(F_BLK, size - start) - drop
            for (c in 0 until MIMI_D) {
                System.arraycopy(
                    out, c * S_BLK + drop * UPS,
                    feat, c * S_DEC + (start + drop) * UPS, keepN * UPS,
                )
            }
            kept = start + drop + keepN
            featPos = kept * UPS
        }

        /** One SEANet window; emits the positions that now have full left context. */
        private fun emitWindow() {
            val model = deconlyW ?: return
            val ins = deconlyWIn ?: return
            val outs = deconlyWOut ?: return
            val start = if (emitted == 0) 0 else emitted - STREAM_L
            val avail = minOf(featPos, start + streamW) - start
            java.util.Arrays.fill(win, 0f)
            for (c in 0 until MIMI_D) {
                System.arraycopy(feat, c * S_DEC + start, win, c * streamW, avail)
            }
            ins[0].writeFloat(win)
            val t0 = System.nanoTime()
            model.run(ins, outs)
            val wav = outs[0].readFloat()
            sSeanet += System.nanoTime() - t0
            val keep = minOf(start + streamW, featPos) - emitted
            val off = (emitted - start) * SPP
            val out = FloatArray(keep * SPP) { wav[off + it].coerceIn(-1f, 1f) }
            emitted += keep
            chunks++
            sAudioChunks++
            onChunk(out)
        }
    }

    /**
     * Streaming synthesis: [onChunk] receives audio as soon as it is decodable
     * instead of after the whole utterance, so playback can start during
     * generation. The concatenated chunks match [synthesize]'s audio to within
     * backend rounding (bit-exact on the host). Falls back to [synthesize] when
     * the streaming SEANet graph is absent.
     */
    fun synthesizeStream(
        text: String,
        voice: String,
        onChunk: (FloatArray) -> Unit,
    ): Result {
        if (deconlyW == null) {
            android.util.Log.w("PocketTTS", "no ${deconlyGraph(streamW)}: one-shot fallback")
            return synthesize(text, voice)
        }
        val t0 = System.nanoTime()
        noiseSeed?.let { rnd.setSeed(it) }
        resetProfile()
        loadVoice(voice)
        val all = ArrayList<FloatArray>()
        var frames = 0
        for (chunk in splitIntoBestSentences(text)) {
            val (prepared, eosGuess) = prepareTextPrompt(chunk)
            val ids = tokenizer.encode(prepared)
            val dec = StreamDecoder { c ->
                if (sFirstChunk < 0) sFirstChunk = (System.nanoTime() - t0) / 1_000_000
                all.add(c)
                onChunk(c)
            }
            val lats = generateChunk(ids, framesAfterEos = eosGuess + 2) { dec.push(it) }
            dec.flush()
            android.util.Log.i(
                "PocketTTS",
                "chunk: ${ids.size} tokens -> ${lats.size} frames, ${dec.chunks} audio chunks",
            )
            frames += lats.size
            sPrompt += ids.size; sFrames += lats.size; sChunks++
        }
        val total = all.sumOf { it.size }
        val out = FloatArray(total)
        var o = 0
        for (a in all) { System.arraycopy(a, 0, out, o, a.size); o += a.size }
        return Result(out, frames, (System.nanoTime() - t0) / 1_000_000, snapshotProfile())
    }

    // ---- text preparation (ports of pocket_tts.models.tts_model) ----------

    /** prepare_text_prompt: normalize whitespace/case/punctuation; guess EOS tail. */
    internal fun prepareTextPrompt(raw: String): Pair<String, Int> {
        var text = raw.trim()
        require(text.isNotEmpty()) { "Text prompt cannot be empty" }
        text = text.replace('\n', ' ').replace('\r', ' ').replace("  ", " ")
        val words = text.trim().split(Regex("\\s+")).size
        val guess = if (words <= 4) 3 else 1
        if (!text[0].isUpperCase()) text = text[0].uppercaseChar() + text.substring(1)
        if (text.last().isLetterOrDigit()) text += "."
        return text to guess
    }

    /** split_into_best_sentences: sentence segments greedily packed <=50 tokens. */
    internal fun splitIntoBestSentences(raw: String): List<String> {
        val (prepared, _) = prepareTextPrompt(raw)
        val tokens = tokenizer.encode(prepared.trim()).toList()

        fun boundaries(list: List<Int>, marks: Set<Int>): List<Int> {
            val idx = ArrayList<Int>()
            idx.add(0)
            var prevWasBoundary = false
            for ((i, tok) in list.withIndex()) {
                if (tok in marks) prevWasBoundary = true
                else {
                    if (prevWasBoundary) idx.add(i)
                    prevWasBoundary = false
                }
            }
            idx.add(list.size)
            return idx
        }

        fun segments(list: List<Int>, idx: List<Int>): List<Pair<Int, String>> =
            (0 until idx.size - 1).map { i ->
                val part = list.subList(idx[i], idx[i + 1])
                part.size to tokenizer.decode(part)
            }

        val sentences = segments(tokens, boundaries(tokens, endTokens))
        val refined = ArrayList<Pair<Int, String>>()
        for ((n, textSeg) in sentences) {
            if (n <= MAX_TOKENS_PER_CHUNK) { refined.add(n to textSeg); continue }
            val sub = tokenizer.encode(textSeg.trim()).toList()
            val subSegs = segments(sub, boundaries(sub, fallbackTokens))
            if (subSegs.size > 1) refined.addAll(subSegs) else refined.add(n to textSeg)
        }

        val chunks = ArrayList<String>()
        var current = ""
        var count = 0
        for ((n, sentence) in refined) {
            when {
                current.isEmpty() -> { current = sentence; count = n }
                count + n > MAX_TOKENS_PER_CHUNK -> {
                    chunks.add(current.trim()); current = sentence; count = n
                }
                else -> { current += " $sentence"; count += n }
            }
        }
        if (current.isNotEmpty()) chunks.add(current.trim())
        return chunks
    }

    override fun close() {
        listOf(lmIn, lmOut, dectxIn, dectxOut, deconlyIn, deconlyOut)
            .forEach { l -> l.forEach { it.close() } }
        lmMsIn?.forEach { it.close() }
        lmMsOut?.forEach { it.close() }
        deconlyWIn?.forEach { it.close() }
        deconlyWOut?.forEach { it.close() }
        lm.close(); lmMs?.close(); dectx.close(); deconly.close()
        deconlyW?.close(); embChannel.close()
        npuEnvironment?.close()
    }

    private fun readF32(f: File): FloatArray {
        val b = f.readBytes()
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.float }
    }
}
