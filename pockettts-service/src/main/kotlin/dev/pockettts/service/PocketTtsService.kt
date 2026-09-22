package dev.pockettts.service

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import dev.pockettts.PocketTtsSession
import dev.pockettts.Voice as TtsVoice
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue

/**
 * Android text-to-speech engine backed by Pocket TTS. Declared by this module's
 * manifest, so including `:pockettts-service` in an app registers it as a system
 * TTS engine; the app only has to provision the models (adb push, bundled
 * assets, or [PocketTtsEngine.ensureModels]).
 *
 * Live audio is generated in the framework's callback thread and written as
 * 16-bit PCM chunks, so playback starts at the first SEANet window rather than
 * after the whole utterance.
 *
 * Locale and voices are ISO-3 throughout: [onGetLanguage] reports `eng`/`USA`,
 * never `en-US`, and the [Voice] objects it publishes carry the stable
 * `pockettts-<name>` IDs alongside the bare engine names.
 *
 * Speech rate and pitch are honoured as independent post-processing (tempo and
 * pitch shift, via `SonicStretcher`), and can be set per request or persisted for
 * the engine through the settings screen.
 */
class PocketTtsService : TextToSpeechService() {

    private var engine: PocketTtsEngine? = null

    @Volatile
    private var current: PocketTtsSession? = null

    /** Voice selected by the last [onLoadVoice]; the request's name wins. */
    @Volatile
    private var selectedVoice: String? = null

    /**
     * The voices the engine *will* be able to speak, resolved from the installed
     * voice files without compiling any graph. The framework queries voices and
     * languages on binder threads while the user browses settings, so this must
     * not be the multi-second engine load.
     */
    private val installedVoices: List<TtsVoice> by lazy {
        val models = PocketTtsModels.default(this)
        val installed = TtsVoice.all().filter { models.store.exists(PocketTts.voiceFile(it.name)) }
        installed.ifEmpty { TtsVoice.all() }
    }

    // ---- lifecycle ---------------------------------------------------------

    @Synchronized
    private fun engine(): PocketTtsEngine {
        engine?.let { return it }
        val e = PocketTtsEngine(this)
        engine = e
        return e
    }

    /**
     * Voice precedence: the request's voice name, then the explicitly loaded
     * voice, then the user's persisted default, then the first installed voice.
     */
    private fun voiceFor(request: SynthesisRequest): TtsVoice {
        val installed = installedVoices
        val wanted = PocketTts.voiceNamed(request.voiceName)
            ?: PocketTts.voiceNamed(selectedVoice)
            ?: PocketTts.voiceNamed(PocketTtsSettings.voice(this, installed.first().name))
        return wanted?.takeIf { w -> installed.any { it.name == w.name } } ?: installed.first()
    }

    // ---- synthesis ---------------------------------------------------------

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        val text = request.charSequenceText?.toString()?.trim().orEmpty()
        if (text.isEmpty()) {
            // Nothing to say; still a valid (empty) utterance.
            callback.start(PocketTts.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1)
            callback.done()
            return
        }

        val voice = voiceFor(request)
        // getSpeechRate()/getPitch() are percentages (100 = natural), sourced from
        // the client's setSpeechRate()/setPitch() or the system default. When a
        // value is exactly the framework default and the user has stored a
        // preference, prefer theirs.
        val requestedRate = request.speechRate
        val rate = (
            if (requestedRate == DEFAULT_RATE) defaultRate() else requestedRate / 100f
            ).coerceIn(PocketTtsSettings.MIN_RATE, PocketTtsSettings.MAX_RATE)
        val requestedPitch = request.pitch
        val pitch = (
            if (requestedPitch == DEFAULT_PITCH) defaultPitch() else requestedPitch / 100f
            ).coerceIn(PocketTtsSettings.MIN_PITCH, PocketTtsSettings.MAX_PITCH)

        val t0 = System.nanoTime()
        android.util.Log.i(
            "PocketTTSTime",
            "service synth: ${text.length} chars voice=${voice.name} rate=$rate pitch=$pitch " +
                "maxBuf=${callback.maxBufferSize}",
        )

        val session: PocketTtsSession
        try {
            session = engine().newSession(voice.name)
        } catch (e: Throwable) {
            android.util.Log.e(TAG, "engine load failed", e)
            // start() must have been called before error() on some frameworks.
            callback.start(PocketTts.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1)
            callback.error(ERROR_MODEL_UNAVAILABLE)
            callback.done()
            return
        }
        session.rate = rate
        session.pitch = pitch
        current = session

        // audioAvailable() must never receive more than this many bytes.
        val out = PcmBuffer(callback.maxBufferSize.coerceAtLeast(2), t0)

        // Generate on a worker and drain the queue here. audioAvailable() blocks
        // while the framework's playback buffer is full; if that happened on the
        // generation thread the LM could not start the next decoder block until
        // the current window had drained, so playback underran between windows.
        // The queue lets the LM run ahead and have the next block ready in time.
        val queue = ArrayBlockingQueue<FloatArray>(QUEUE_CHUNKS)
        val sentinel = FloatArray(0)
        var producerError: Throwable? = null
        val producer = Thread({
            try {
                session.stream(text) { queue.put(it) }
            } catch (e: Throwable) {
                producerError = e
            } finally {
                // Always unblock the consumer, even after cancel or failure.
                while (true) {
                    try {
                        queue.put(sentinel)
                        break
                    } catch (_: InterruptedException) {
                        Thread.currentThread().interrupt()
                    }
                }
            }
        }, "pockettts-tts").apply { isDaemon = true }

        var failed = false
        var firstAudio = -1L
        try {
            if (callback.start(PocketTts.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1)
                == TextToSpeech.STOPPED
            ) {
                return
            }
            producer.start()
            var stopped = false
            while (true) {
                val chunk = queue.take()
                if (chunk === sentinel) break
                if (stopped) continue
                if (firstAudio < 0) {
                    firstAudio = (System.nanoTime() - t0) / 1_000_000
                    android.util.Log.i(
                        "PocketTTSTime",
                        "service first audio at ${firstAudio}ms (${chunk.size} samples)",
                    )
                }
                // The framework stops calling back once the utterance is
                // stopped or done; stop generating then, but keep draining so
                // the producer can reach its sentinel.
                if (callback.hasFinished() == true || !out.put(chunk, callback)) {
                    session.cancel()
                    stopped = true
                }
            }
            producerError?.let { throw it }
            android.util.Log.i(
                "PocketTTSTime",
                "service done: ${(System.nanoTime() - t0) / 1_000_000}ms firstAudio=${firstAudio}ms",
            )
        } catch (e: Throwable) {
            failed = true
            session.cancel()
            android.util.Log.e(TAG, "synthesis failed", e)
            // Let the producer finish so the join below cannot hang.
            runCatching { while (queue.take() !== sentinel) { /* drain */ } }
        } finally {
            current = null
            producer.join(5_000)
            // done() is mandatory once start() succeeded, errors included.
            callback.done()
            if (failed) callback.error(ERROR_SYNTHESIS)
        }
    }

    /**
     * Converts decoded (and rate-shaped) float PCM into 16-bit blocks the
     * framework accepts. Every block of at most
     * [SynthesisCallback.getMaxBufferSize] bytes is handed to
     * [SynthesisCallback.audioAvailable] as soon as it is assembled, so nothing
     * is held back waiting for a full buffer.
     */
    private class PcmBuffer(sizeBytes: Int, private val t0: Long) {
        // A sample is 2 bytes. Keep the buffer even so every block is whole
        // samples; an odd framework max would otherwise strand the last byte.
        private val buf = ByteArray(sizeBytes - sizeBytes % 2)
        private var calls = 0
        private var bytes = 0L

        /** Feed [audio]; returns false when the framework stopped the utterance. */
        fun put(audio: FloatArray, callback: SynthesisCallback): Boolean {
            var i = 0
            while (i < audio.size) {
                val take = minOf(buf.size / 2, audio.size - i)
                var o = 0
                for (j in i until i + take) {
                    val s = (audio[j].coerceIn(-1f, 1f) * 32767f).toInt()
                    buf[o++] = (s and 0xFF).toByte()
                    buf[o++] = ((s shr 8) and 0xFF).toByte()
                }
                i += take
                val stopped = callback.audioAvailable(buf, 0, o) == TextToSpeech.STOPPED
                bytes += o
                android.util.Log.i(
                    "PocketTTSTime",
                    "  audioAvailable#${++calls} ${o}B total=${bytes}B " +
                        "at ${(System.nanoTime() - t0) / 1_000_000}ms",
                )
                if (stopped) return false
            }
            return true
        }
    }

    override fun onStop() {
        current?.cancel()
    }

    // ---- language ----------------------------------------------------------

    /**
     * ISO-3 language, ISO-3 country, variant - *not* the `en-US`-style tag.
     * Only called on API <= 17, but it must stay consistent with
     * [onIsLanguageAvailable] and [onLoadLanguage].
     */
    override fun onGetLanguage(): Array<String> = arrayOf(LANG, COUNTRY, "")

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int =
        languageStatus(lang, country)

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int =
        languageStatus(lang, country)

    /**
     * `eng`, `eng`/`USA` and English-region locales are supported; everything
     * else is `LANG_NOT_SUPPORTED`. The return has to be consistent with
     * [onLoadLanguage], and it is what `onGetVoices` filters on, so a locale
     * without an ISO-3 code (or a non-English one) never becomes a voice.
     */
    private fun languageStatus(lang: String?, country: String?): Int {
        val language = lang?.trim()?.lowercase().orEmpty()
        if (language != LANG_ALPHA2 && language != LANG) return TextToSpeech.LANG_NOT_SUPPORTED
        val c = country?.trim()?.lowercase().orEmpty()
        return when {
            c.isEmpty() -> TextToSpeech.LANG_AVAILABLE
            c == COUNTRY.lowercase() -> TextToSpeech.LANG_COUNTRY_AVAILABLE
            // "GB"/"AUS"/... speak the same model; accept rather than hide them.
            englishCountry(c) -> TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE
            else -> TextToSpeech.LANG_NOT_SUPPORTED
        }
    }

    private fun englishCountry(iso3: String): Boolean =
        iso3 in ENGLISH_COUNTRIES || (iso3.length == 2 && iso3.uppercase(Locale.US) in ENGLISH_COUNTRIES)

    // ---- voices ------------------------------------------------------------

    override fun onGetVoices(): MutableList<Voice> = installedVoices.map { it.toFrameworkVoice() }
        .toMutableList()

    override fun onIsValidVoiceName(voiceName: String?): Int =
        if (PocketTts.voiceNamed(voiceName) != null) TextToSpeech.SUCCESS else TextToSpeech.ERROR

    override fun onLoadVoice(voiceName: String?): Int {
        val v = PocketTts.voiceNamed(voiceName) ?: return TextToSpeech.ERROR
        selectedVoice = v.name
        return TextToSpeech.SUCCESS
    }

    /** The voice `setLanguage` should resolve to: `alba`, the first bundled voice. */
    override fun onGetDefaultVoiceNameFor(
        lang: String?,
        country: String?,
        variant: String?,
    ): String = installedVoices.first().id

    override fun onGetFeaturesForLanguage(
        lang: String?,
        country: String?,
        variant: String?,
    ): MutableSet<String> = mutableSetOf()

    private fun TtsVoice.toFrameworkVoice(): Voice = Voice(
        id,
        locale,
        quality,
        latency,
        false, // fully on-device
        features,
    )

    // ---- defaults / settings ----------------------------------------------

    /**
     * The voice to speak when a client has not chosen one (e.g. `setLanguage`),
     * persisted by [PocketTtsSettingsActivity]. Functionally the same as
     * [onGetDefaultVoiceNameFor]; kept for the settings UI.
     */
    fun defaultVoice(): String = PocketTtsSettings.voice(this, installedVoices.first().name)

    fun setDefaultVoice(name: String) = PocketTtsSettings.setVoice(this, name)

    /** Persisted speech rate (1.0 = natural), applied when a request has none. */
    fun defaultRate(): Float = PocketTtsSettings.rate(this)

    fun setDefaultRate(rate: Float) = PocketTtsSettings.setRate(this, rate)

    /** Persisted voice pitch (1.0 = natural), applied when a request has none. */
    fun defaultPitch(): Float = PocketTtsSettings.pitch(this)

    fun setDefaultPitch(pitch: Float) = PocketTtsSettings.setPitch(this, pitch)

    override fun onDestroy() {
        engine?.close()
        engine = null
        super.onDestroy()
    }

    companion object {
        private const val TAG = "PocketTTSService"

        /** ISO-639-2/T (B) language, the code the framework speaks in. */
        const val LANG = "eng"
        private const val LANG_ALPHA2 = "en"
        /** ISO-3166-1 alpha-3 for the United States. */
        const val COUNTRY = "USA"
        private val ENGLISH_COUNTRIES = setOf(
            "USA", "GBR", "AUS", "CAN", "NZL", "IRL", "ZAF", "IND", "SGP", "PHL",
        )

        /** Callback error codes, from [SynthesisCallback]. */
        private const val ERROR_SYNTHESIS = -1
        private const val ERROR_MODEL_UNAVAILABLE = -2

        /** `TextToSpeech.Engine.DEFAULT_RATE`, the percentage "no preference" value. */
        private const val DEFAULT_RATE = 100

        /** `TextToSpeech.Engine.DEFAULT_PITCH`, the percentage "no preference" value. */
        private const val DEFAULT_PITCH = 100

        /**
         * Decoded chunks the producer may run ahead of the framework. Two would
         * cover the decode gap; a few more absorb jitter without much memory.
         */
        private const val QUEUE_CHUNKS = 8
    }
}
