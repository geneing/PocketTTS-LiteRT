package dev.pockettts.service

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import dev.pockettts.Pcm
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsSession

/**
 * Android text-to-speech engine backed by Pocket TTS. Declared by this module's
 * manifest, so including `:pockettts-service` in an app registers it as a system
 * TTS engine; the app only has to provision the models (adb push, bundled
 * assets, or [PocketTtsEngine.ensureModels]).
 *
 * Live audio is generated in the framework's callback thread and written as
 * 16-bit PCM chunks, so playback starts at the first SEANet window rather than
 * after the whole utterance.
 */
class PocketTtsService : TextToSpeechService() {

    private var engine: PocketTtsEngine? = null

    @Volatile
    private var current: PocketTtsSession? = null

    @Synchronized
    private fun engine(): PocketTtsEngine {
        engine?.let { return it }
        val e = PocketTtsEngine(this)
        engine = e
        return e
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        val text = request.charSequenceText?.toString()?.trim().orEmpty()
        if (text.isEmpty()) {
            callback.start(PocketTts.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1)
            callback.done()
            return
        }
        val voice = request.voiceName?.lowercase()?.takeIf { it in PocketTts.VOICES }
            ?: PocketTts.VOICES.first()
        try {
            val session = engine().newSession(voice)
            current = session
            callback.start(PocketTts.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1)
            session.stream(text) { chunk ->
                val pcm = Pcm.floatToPcm16(chunk)
                callback.audioAvailable(pcm, 0, pcm.size)
            }
            callback.done()
        } catch (e: Throwable) {
            android.util.Log.e("PocketTTSService", "synthesis failed", e)
            callback.error()
        } finally {
            current = null
        }
    }

    override fun onStop() {
        current?.cancel()
    }

    override fun onGetLanguage(): Array<String> = arrayOf("eng", "USA", "")

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int =
        TextToSpeech.LANG_AVAILABLE

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int =
        TextToSpeech.LANG_AVAILABLE

    override fun onDestroy() {
        engine?.close()
        engine = null
        super.onDestroy()
    }
}
