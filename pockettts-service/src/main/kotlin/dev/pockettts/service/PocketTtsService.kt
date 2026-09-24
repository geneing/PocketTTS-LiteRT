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
    private var destroyed = false

    @Volatile
    private var current: PocketTtsSession? = null

    /** Voice selected by the last [onLoadVoice]; the request's name wins. */
    @Volatile
    private var selectedVoice: String? = null

    /**
     * The voices the engine *will* be able to speak, resolved from the installed
     * voice files without compiling any graph. Voice/language queries stay
     * side-effect-free; graph loading is handled once during service startup.
     */
    private val installedVoices: List<TtsVoice> by lazy {
        val models = PocketTtsModels.default(this)
        val installed = TtsVoice.all().filter { models.store.exists(PocketTts.voiceFile(it.name)) }
        installed.ifEmpty { TtsVoice.all() }
    }

    // ---- lifecycle ---------------------------------------------------------

    override fun onCreate() {
        super.onCreate()
        // Compile/load the LiteRT graphs before the first utterance. Do the work
        // off the service main thread, but finish it before the framework can
        // bind and submit synthesis requests; otherwise the first request pays
        // this multi-second startup cost before it can receive even its first PCM.
        val preload = Thread({
            val started = System.nanoTime()
            try {
                engine()
                android.util.Log.i(
                    "PocketTTSTime",
                    "service engine preload ready in ${(System.nanoTime() - started) / 1_000_000}ms",
                )
            } catch (e: Throwable) {
                android.util.Log.e(TAG, "engine preload failed; synthesis will retry lazily", e)
            }
        }, "pockettts-engine-preload").apply { isDaemon = true }
        preload.start()
        try {
            preload.join()
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    @Synchronized
    private fun engine(): PocketTtsEngine {
        check(!destroyed) { "TTS service is shutting down" }
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
        val utt = UTTERANCE.incrementAndGet()
        android.util.Log.i(
            "PocketTTSTime",
            "service synth#$utt TRACE enter text(${text.length})=\"${preview(text)}\" " +
                "voice=${voice.name} rate=$rate pitch=$pitch maxBuf=${callback.maxBufferSize} " +
                "@${android.os.SystemClock.elapsedRealtime()}",
        )

        val session: PocketTtsSession
        try {
            val loadT = System.nanoTime()
            session = engine().newSession(voice.name)
            val loadMs = (System.nanoTime() - loadT) / 1_000_000
            if (loadMs >= 30) {
                android.util.Log.i(
                    "PocketTTSTime",
                    "service #$utt TRACE engine+session load=${loadMs}ms " +
                        "t=${(System.nanoTime() - t0) / 1_000_000}ms",
                )
            }
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
            var produced = 0
            android.util.Log.i(
                "PocketTTSTime",
                "service #$utt TRACE producer start t=${(System.nanoTime() - t0) / 1_000_000}ms",
            )
            try {
                session.stream(text) { chunk ->
                    val putT = System.nanoTime()
                    queue.put(chunk)
                    val putMs = (System.nanoTime() - putT) / 1_000_000
                    produced++
                    android.util.Log.i(
                        "PocketTTSTime",
                        "service #$utt TRACE producer put#$produced ${chunk.size} samples " +
                            "blocked=${putMs}ms queue=${queue.size}/$QUEUE_CHUNKS " +
                            "t=${(System.nanoTime() - t0) / 1_000_000}ms",
                    )
                }
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
        var producerStarted = false
        var firstAudio = -1L
        try {
            if (callback.start(PocketTts.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1)
                == TextToSpeech.STOPPED
            ) {
                return
            }
            producer.start()
            producerStarted = true
            var stopped = false
            var consumed = 0
            while (true) {
                val takeT = System.nanoTime()
                val chunk = queue.take()
                val takeMs = (System.nanoTime() - takeT) / 1_000_000
                if (chunk === sentinel) break
                if (stopped) continue
                consumed++
                android.util.Log.i(
                    "PocketTTSTime",
                    "service #$utt TRACE consumer chunk#$consumed waited=${takeMs}ms " +
                        "queue=${queue.size}/$QUEUE_CHUNKS t=${(System.nanoTime() - t0) / 1_000_000}ms",
                )
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
            if (producerStarted) {
                runCatching { while (queue.take() !== sentinel) { /* drain */ } }
            }
        } finally {
            current = null
            try {
                if (producerStarted) producer.join(5_000)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            } finally {
                try {
                    if (failed) callback.error(ERROR_SYNTHESIS)
                } finally {
                    // done() closes the callback on success and failure; errors
                    // must be reported before the terminal done notification.
                    callback.done()
                }
            }
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
                val callT = System.nanoTime()
                val stopped = callback.audioAvailable(buf, 0, o) == TextToSpeech.STOPPED
                val callMs = (System.nanoTime() - callT) / 1_000_000
                bytes += o
                android.util.Log.i(
                    "PocketTTSTime",
                    "  audioAvailable#${++calls} ${o}B total=${bytes}B " +
                        "clientCall=${callMs}ms at ${(System.nanoTime() - t0) / 1_000_000}ms",
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
        val toClose = synchronized(this) {
            destroyed = true
            engine.also { engine = null }
        }
        toClose?.close()
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

        /** Monotonic id per [onSynthesizeText], so log lines can be correlated. */
        private val UTTERANCE = java.util.concurrent.atomic.AtomicLong()

        /** Cap a text payload so a single log line stays readable. */
        internal fun preview(s: String, max: Int = 1200): String =
            if (s.length <= max) s else s.substring(0, max) + "…(+" + (s.length - max) + " chars)"
    }
}
