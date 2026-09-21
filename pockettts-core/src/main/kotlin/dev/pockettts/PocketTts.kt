package dev.pockettts

import java.io.File

/**
 * Pocket TTS (Kyutai, ~100M) — a flow-matching language model over continuous
 * 32-dim Mimi latents, plus a tiny Mimi codec, running on LiteRT
 * `CompiledModel`.
 *
 * This object holds the fixed model geometry and the file names the engine
 * expects from a [ModelSource].
 */
object PocketTts {

    // ---- flow-LM geometry -------------------------------------------------
    const val H = 1024               // model width
    const val HD = 64                // head dim
    const val NH = 16                // heads
    const val LAYERS = 6
    const val G = LAYERS * NH        // packed KV groups
    const val PMAX = 512             // KV capacity: voice + text + audio frames
    const val LDIM = 32              // Mimi latent dim
    const val THETA = 10000.0

    // ---- Mimi decoder geometry -------------------------------------------
    const val UPS = 16               // 12.5 Hz -> 200 Hz
    const val MIMI_D = 512
    const val F_BLK = 64             // dec_tx block payload frames
    const val F_HOP = 32             // dec_tx block hop
    const val S_BLK = F_BLK * UPS
    const val DEC_FRAMES = 256       // one-shot deconly window frames
    const val S_DEC = DEC_FRAMES * UPS
    const val SPF = 1920             // samples per 12.5 Hz frame
    const val SPP = SPF / UPS        // samples per Mimi feature position
    const val SAMPLE_RATE = 24000

    // ---- generation defaults (english config / pocket_tts defaults) ------
    const val TEMP = 0.3f
    const val EOS_THRESHOLD = -4.0f
    const val MAX_TOKENS_PER_CHUNK = 50
    const val TOKENS_PER_SECOND = 3.0
    const val GEN_SECONDS_PADDING = 2.0
    const val FRAME_RATE = 12.5
    const val MASK_NEG = -1e4f

    // ---- graph file names -------------------------------------------------
    /** Step + flow head fused, fp16. */
    const val LM = "pt_flowlm_fused_fp16.tflite"

    /**
     * Dynamic-range int8 flow-LM: int8 weights, fp32 activations, so the host
     * protocol is unchanged. ~2.3x faster per frame than [LM] on XNNPACK.
     */
    const val LM_INT8 = "pt_flowlm_fused_dyn8_all.tflite"

    /** int8 when it has been pushed/provisioned, else fp16. */
    fun lmGraphFor(dir: File): String = if (File(dir, LM_INT8).exists()) LM_INT8 else LM

    /** N-step fused decode graph: N frames per invocation. */
    fun msGraph(n: Int) = "pt_flowlm_ms${n}_fp16.tflite"

    const val DEC_TX = "pt_mimi_dec_tx_fp16.tflite"

    /** AOT-compiled Tensor G5 variant of a stock graph (NPU placement). */
    fun g5Variant(name: String) = name.replace(".tflite", "_g5.tflite")

    /** One-shot SEANet decoder. */
    const val DECONLY = "pt_mimi_deconly_fp16.tflite"

    /** Smaller-window SEANet decoder, for streaming. */
    fun deconlyGraph(w: Int) = "pt_mimi_deconly_w${w}_fp16.tflite"

    /** Default SEANet window (feature positions) for streaming. */
    const val STREAM_W = 512

    /** Left context kept when sliding the SEANet window; margin over the ~8 position RF. */
    const val STREAM_L = 16

    const val EMBED = "pt_embed_f16.bin"
    const val INPUT_LINEAR = "pt_input_linear_f32.bin"
    const val BOS = "pt_bos_input_f32.bin"
    const val NEUTRAL = "pt_neutral_latent_f32.bin"
    const val TOKENIZER = "pt_tokenizer.tsv"

    fun voiceFile(name: String) = "pt_voice_$name.bin"

    /** Locale voices bundled with the model (CC-BY-4.0 / CC0 only). */
    val VOICES = listOf("alba", "marius", "javert", "charles", "mary", "eve")
}

/** Stage timings for one synthesis call (ms unless noted). */
data class TtsProfile(
    val lmSteps: Int,
    val promptSteps: Int,
    val genFrames: Int,
    val lmInMs: Long,
    val lmRunMs: Long,
    val lmReadMs: Long,
    val decTxMs: Long,
    val seanetMs: Long,
    val chunks: Int,
    /** LM graph invocations: == lmSteps for the 1-step graph, fewer for N-step. */
    val lmInvocations: Int = 0,
    val lmInBytes: Long = 0,
    val lmOutBytes: Long = 0,
    /** Time to the first audio chunk (streaming), or -1 for one-shot. */
    val firstChunkMs: Long = -1,
    val audioChunks: Int = 0,
)

/** One utterance's result. [audio] is 24 kHz mono float PCM, [ms] is wall clock. */
data class TtsResult(
    val audio: FloatArray,
    val frames: Int,
    val ms: Long,
    val profile: TtsProfile,
)

/** Receives streaming audio and terminal events. Called on the synth thread. */
interface SpeechListener {
    fun onAudio(chunk: FloatArray)
    fun onDone(result: TtsResult)
    fun onError(error: Throwable)
}

/** Handle for an in-flight asynchronous utterance. */
interface Utterance {
    /** Barge-in: stop generating and discard queued audio as soon as possible. */
    fun cancel()
}
