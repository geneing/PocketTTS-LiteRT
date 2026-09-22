package dev.pockettts

import android.content.Context
import android.util.Log
import sonic.Sonic
import java.io.File

/**
 * Everything that shapes an engine: where the files come from, where each graph
 * runs, and the decode knobs. Immutable; one config per [PocketTtsEngine].
 */
class PocketTtsConfig(
    /** Where model files are resolved from. */
    val models: PocketTtsModels,
    /** Per-graph accelerator choice. */
    val placement: Placement,
    /** Override the flow-LM graph filename (e.g. a quantized variant). null = auto. */
    val lmGraph: String? = null,
    /** Frames per LM invocation; >1 needs the matching `pt_flowlm_ms{N}` graph. */
    val lmSteps: Int = 1,
    /** SEANet window (feature positions) for streaming. */
    val streamW: Int = PocketTts.STREAM_W,
    /** When set, every synthesis reseeds the noise RNG so repeats are identical. */
    val noiseSeed: Long? = null,
    /** When set, GPU graphs serialize their compiled program cache here. */
    val gpuCache: File? = null,
    /**
     * The voices this engine offers, defaulting to every bundled voice whose
     * `.bin` is actually resolvable through [models]. Narrow it to hide voices;
     * the first entry is the default. An empty list is never returned: at least
     * one voice file must be present or synthesis cannot work at all.
     */
    val voices: List<Voice> = defaultVoices(models),
) {
    init {
        require(voices.isNotEmpty()) { "a config needs at least one voice" }
    }

    companion object {
        /** Every bundled voice whose state file is installed, else all of them. */
        fun defaultVoices(models: PocketTtsModels): List<Voice> {
            val installed = Voice.all().filter { models.store.exists(PocketTts.voiceFile(it.name)) }
            return installed.ifEmpty { Voice.all() }
        }

        /**
         * The device policy: adb-pushed models first, GitHub release fallback,
         * and [Placement.default] for the accelerator split.
         */
        fun default(
            context: Context,
            models: PocketTtsModels = PocketTtsModels.default(context),
        ): PocketTtsConfig {
            val dir = context.getExternalFilesDir(null) ?: context.filesDir
            return PocketTtsConfig(models, Placement.default(context, dir))
        }
    }
}

/**
 * Post-generation speech shaping: independent tempo (rate) and pitch, applied to
 * the decoded 24 kHz float PCM. It is a separate pass, so the LM/Mimi graphs
 * never see a different sampling rate.
 *
 * Backed by the vendored [sonic.Sonic] library (see docs/library.md for the
 * attribution): `rate` is a constant-pitch time-stretch, `pitch` a pitch shift
 * that keeps the duration. They compose, so any mixture works and `1f, 1f` is an
 * exact pass-through.
 *
 * @return the shaped audio; the input itself when both are 1.
 */
fun sonicStretch(audio: FloatArray, rate: Float, pitch: Float = 1f): FloatArray {
    if (audio.isEmpty()) return audio
    val s = SonicStretcher(rate, pitch)
    val out = s.push(audio)
    val tail = s.finish()
    return if (tail.isEmpty()) out else out + tail
}

/**
 * Streaming wrapper over [sonic.Sonic]. Feed decoded chunks with [push]; the
 * result grows as input arrives. Call [finish] once at the end for the tail.
 *
 * Mapping: Sonic's stream input runs at `s = speed/pitch` (time-stretch) and then
 * `r = rate*pitch` (resample) when chord pitch is off. Setting `speed = rate`,
 * `pitch = pitch`, `rate = 1` therefore gives a total speed of `rate` and a total
 * pitch of `pitch`, independently.
 *
 * Sonic is internally 16-bit, so anything other than the identity path is
 * requantized. It buffers ~31 ms (`2*maxPeriod`) before producing output, so
 * [push] can legitimately return nothing.
 */
class SonicStretcher(
    rate: Float,
    pitch: Float,
    private val sampleRate: Int = PocketTts.SAMPLE_RATE,
) {
    private val speed = rate.coerceIn(MIN_SPEED, MAX_SPEED)
    private val shifted = pitch.coerceIn(MIN_PITCH, MAX_PITCH)
    private val passthrough = speed == 1f && shifted == 1f

    private val stream: Sonic? = if (passthrough) {
        null
    } else {
        Sonic(sampleRate, 1).apply {
            // Qualified: inside apply, `speed` would resolve to Sonic's own
            // synthetic getSpeed()/setSpeed() property and set nothing.
            setSpeed(this@SonicStretcher.speed)
            setPitch(this@SonicStretcher.shifted)
            setRate(1f)
            setChordPitch(false)
            setQuality(0)
        }
    }

    /** Reused read buffer; [read] fills it and copies out only what it got. */
    private val scratch = FloatArray(SCRATCH)

    // ---- timing (see [stats]) --------------------------------------------
    private var pushes = 0
    internal var sonicMs = 0L
        private set
    internal var inSamples = 0L
        private set
    internal var outSamples = 0L
        private set

    /** Shape [chunk]; returns it unchanged on the identity path. */
    fun push(chunk: FloatArray): FloatArray {
        val s = stream ?: return chunk
        if (chunk.isEmpty()) return chunk
        val t = System.nanoTime()
        s.writeFloatToStream(chunk, chunk.size)
        val out = read(s)
        val ms = (System.nanoTime() - t) / 1_000_000
        sonicMs += ms
        inSamples += chunk.size
        outSamples += out.size
        pushes++
        Log.i(
            "PocketTTSTime",
            "sonic push#$pushes in=${chunk.size} out=${out.size} ${ms}ms " +
                "(speed=$speed pitch=$shifted)",
        )
        return out
    }

    /** One-line aggregate for the end of an utterance. */
    fun stats(): String =
        if (passthrough) {
            "sonic passthrough"
        } else {
            "sonic ${pushes} pushes ${sonicMs}ms in=$inSamples out=$outSamples " +
                "(speed=$speed pitch=$shifted)"
        }

    /** Flush Sonic's internal buffers and return the tail. */
    fun finish(): FloatArray {
        val s = stream ?: return FloatArray(0)
        s.flushStream()
        return read(s)
    }

    /**
     * Drain everything Sonic currently has. Sonic reports the total up front, so
     * the result is sized once instead of concatenating per read (which is
     * quadratic when a chunk yields many buffers).
     */
    private fun read(s: Sonic): FloatArray {
        val total = s.samplesAvailable()
        if (total <= 0) return FloatArray(0)
        val out = FloatArray(total)
        var o = 0
        while (o < total) {
            val n = s.readFloatFromStream(scratch, minOf(scratch.size, total - o))
            if (n <= 0) break
            System.arraycopy(scratch, 0, out, o, n)
            o += n
        }
        return if (o == total) out else out.copyOf(o)
    }

    companion object {
        const val MIN_SPEED = 0.5f
        const val MAX_SPEED = 2.0f
        const val MIN_PITCH = 0.5f
        const val MAX_PITCH = 2.0f

        /** Sonic's read buffer is caller-owned; 4096 is ample per drain call. */
        private const val SCRATCH = 4096
    }
}
