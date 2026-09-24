package com.pockettts

import android.content.Context
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * End-to-end check of the TTS engine through the real `android.speech.tts`
 * client, on a device: the service binds, publishes voices and an ISO-3
 * language, and honours `setSpeechRate`/`setPitch` independently.
 *
 * Rate is verified by duration (tempo) at constant measured pitch; pitch by the
 * measured fundamental at constant duration; the two compose. Numbers are
 * logged under tag `TtsVerify` for the record.
 */
@RunWith(AndroidJUnit4::class)
class TtsServiceInstrumentedTest {

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun connect(): TextToSpeech {
        val ready = CountDownLatch(1)
        var status = -1
        val tts = TextToSpeech(ctx, { status = it; ready.countDown() }, ENGINE)
        assertTrue("TTS init timed out", ready.await(60, TimeUnit.SECONDS))
        assertEquals("TTS init status", TextToSpeech.SUCCESS, status)
        Log.i(TAG, "requested=$ENGINE systemDefault=${tts.defaultEngine}")
        return tts
    }

    private fun synth(tts: TextToSpeech, rate: Float, pitch: Float, tag: String): File {
        tts.setSpeechRate(rate)
        tts.setPitch(pitch)
        val file = File(ctx.filesDir, "verify_$tag.wav")
        file.delete()
        val done = CountDownLatch(1)
        val errors = ConcurrentHashMap<String, String>()
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) = Unit
            override fun onDone(utteranceId: String?) { done.countDown() }
            @Deprecated("deprecated in Java")
            override fun onError(utteranceId: String?) { errors[tag] = "error"; done.countDown() }
            override fun onError(utteranceId: String?, errorCode: Int) {
                errors[tag] = "error $errorCode"; done.countDown()
            }
        })
        val rc = tts.synthesizeToFile(TEXT, null, file, tag)
        assertEquals("synthesizeToFile rc ($tag)", TextToSpeech.SUCCESS, rc)
        assertTrue("synthesis timed out ($tag)", done.await(120, TimeUnit.SECONDS))
        assertEquals("synthesis error ($tag)", null, errors[tag])
        assertTrue("empty output ($tag)", file.length() > 44)
        return file
    }

    @Test
    fun voicesAndLanguage() {
        val tts = connect()
        try {
            val voice = waitForPocketTtsVoice(tts)
            val ids = tts.voices!!.map { it.name }
            Log.i(TAG, "voices=${ids.size} $ids")
            assertTrue("pockettts voice ids", ids.any { it.startsWith("pockettts-") })
            assertTrue("selected pockettts voice", voice.name.startsWith("pockettts-"))

            val usa = tts.isLanguageAvailable(Locale("eng", "USA"))
            val eng = tts.isLanguageAvailable(Locale("eng"))
            Log.i(TAG, "lang eng/USA=$usa eng=$eng")
            assertTrue("eng/USA available", usa >= 0)
            assertTrue("eng available", eng >= 0)
        } finally {
            tts.shutdown()
        }
    }

    /** The engine's voices load asynchronously; wait for a `pockettts-*` id. */
    private fun waitForPocketTtsVoice(tts: TextToSpeech): android.speech.tts.Voice {
        val deadline = System.currentTimeMillis() + 30_000
        while (System.currentTimeMillis() < deadline) {
            val v = tts.voices?.firstOrNull { it.name.startsWith("pockettts-") }
            if (v != null) return v
            Thread.sleep(200)
        }
        throw AssertionError("no pockettts voice; engine=${tts.defaultEngine} voices=${tts.voices}")
    }

    /**
     * Diagnostic: speak through the framework (real AudioTrack playback, so
     * `audioAvailable` can block) and log when chunks are produced vs consumed
     * under the `PocketTTSTime` tag.
     */
    @Test
    fun speakTiming() {
        val tts = connect()
        try {
            tts.voice = waitForPocketTtsVoice(tts)
            tts.setSpeechRate(1.5f)
            val done = CountDownLatch(1)
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) = Unit
                override fun onDone(utteranceId: String?) { done.countDown() }
                @Deprecated("deprecated in Java")
                override fun onError(utteranceId: String?) { done.countDown() }
                override fun onError(utteranceId: String?, errorCode: Int) { done.countDown() }
            })
            tts.speak(SPEAK_TEXT, TextToSpeech.QUEUE_FLUSH, null, "speak")
            assertTrue("speak timed out", done.await(120, TimeUnit.SECONDS))
        } finally {
            tts.shutdown()
        }
    }

    /** Real AudioTrack playback, submitting each sentence after the previous onDone. */
    @Test
    fun speakFourSentencesSequentially() {
        val tts = connect()
        try {
            val voice = tts.voices?.firstOrNull { it.name.equals("pockettts-alba", ignoreCase = true) }
                ?: waitForPocketTtsVoice(tts)
            tts.voice = voice
            tts.setSpeechRate(1.4f)
            tts.setPitch(1.0f)

            val sentences = listOf(
                "Its car parts are sized in metric units.",
                "So are its bicycles.",
                "It is no longer enough for American exporters simply to label their products in both " +
                    "American and metric units (soft metric); trade groups abroad are demanding that " +
                    "goods be delivered in even metric units (hard metric).",
                "Oddly, as more Americans are lured into using the metric system, it may be that the " +
                    "nation will lose the very uniformity of weights and measures that has long made " +
                    "the metric system seem unnecessary in the United States.",
            )
            val submittedAt = ConcurrentHashMap<String, Long>()
            val firstAudioAt = ConcurrentHashMap<String, Long>()
            val doneAt = ConcurrentHashMap<String, Long>()
            val doneById = ConcurrentHashMap<String, CountDownLatch>()
            val errors = ConcurrentHashMap<String, String>()
            val sequenceAt = SystemClock.elapsedRealtime()

            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    val id = utteranceId ?: return
                    val now = SystemClock.elapsedRealtime()
                    Log.i(
                        TAG,
                        "sequence onStart $id at=${now - sequenceAt}ms " +
                            "afterSubmit=${now - (submittedAt[id] ?: now)}ms",
                    )
                }

                override fun onBeginSynthesis(
                    utteranceId: String?,
                    sampleRateInHz: Int,
                    audioFormat: Int,
                    channelCount: Int,
                ) {
                    val id = utteranceId ?: return
                    val now = SystemClock.elapsedRealtime()
                    Log.i(
                        TAG,
                        "sequence onBegin $id at=${now - sequenceAt}ms " +
                            "afterSubmit=${now - (submittedAt[id] ?: now)}ms " +
                            "rate=$sampleRateInHz format=$audioFormat channels=$channelCount",
                    )
                }

                override fun onAudioAvailable(utteranceId: String?, audio: ByteArray) {
                    val id = utteranceId ?: return
                    val now = SystemClock.elapsedRealtime()
                    if (firstAudioAt.putIfAbsent(id, now) == null) {
                        Log.i(
                            TAG,
                            "sequence firstAudio $id at=${now - sequenceAt}ms " +
                                "afterSubmit=${now - (submittedAt[id] ?: now)}ms bytes=${audio.size}",
                        )
                    }
                }

                override fun onDone(utteranceId: String?) {
                    val id = utteranceId ?: return
                    val now = SystemClock.elapsedRealtime()
                    val submitted = submittedAt[id] ?: now
                    val firstAudio = firstAudioAt[id] ?: -1L
                    Log.i(
                        TAG,
                        "sequence onDone $id at=${now - sequenceAt}ms " +
                            "afterSubmit=${now - submitted}ms " +
                            "afterFirstAudio=${if (firstAudio < 0) -1 else now - firstAudio}ms",
                    )
                    doneAt[id] = now
                    doneById[id]?.countDown()
                }

                @Deprecated("deprecated in Java")
                override fun onError(utteranceId: String?) {
                    val id = utteranceId ?: "unknown"
                    errors[id] = "error"
                    Log.e(TAG, "sequence onError $id")
                    doneById[id]?.countDown()
                }

                @Deprecated("deprecated in Java")
                override fun onError(utteranceId: String?, errorCode: Int) {
                    val id = utteranceId ?: "unknown"
                    errors[id] = "error $errorCode"
                    Log.e(TAG, "sequence onError $id code=$errorCode")
                    doneById[id]?.countDown()
                }
            })

            Log.i(TAG, "sequence using engine=$ENGINE voice=${voice.name} rate=1.4 pitch=1.0")
            var previousDoneAt = sequenceAt
            for ((index, sentence) in sentences.withIndex()) {
                val id = "evie-seq-$index"
                val submitted = SystemClock.elapsedRealtime()
                submittedAt[id] = submitted
                val utteranceDone = CountDownLatch(1)
                doneById[id] = utteranceDone
                Log.i(
                    TAG,
                    "sequence submit $id at=${submitted - sequenceAt}ms " +
                        "afterPreviousDone=${submitted - previousDoneAt}ms " +
                        "chars=${sentence.length}",
                )
                val queueMode = if (index == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
                assertEquals("speak failed for $id", TextToSpeech.SUCCESS, tts.speak(sentence, queueMode, null, id))
                assertTrue("$id playback timed out", utteranceDone.await(120, TimeUnit.SECONDS))
                assertTrue("TTS error for $id: ${errors[id]}", errors[id] == null)
                previousDoneAt = doneAt[id] ?: SystemClock.elapsedRealtime()
            }
            assertTrue("TTS errors: $errors", errors.isEmpty())
            Log.i(TAG, "sequence playback complete total=${SystemClock.elapsedRealtime() - sequenceAt}ms")
        } finally {
            tts.shutdown()
        }
    }

    @Test
    fun rateAndPitchAreIndependent() {
        val tts = connect()
        try {
            val voice = waitForPocketTtsVoice(tts)
            Log.i(TAG, "using voice=${voice.name}")
            tts.voice = voice
            val base = readWav(synth(tts, 1.0f, 1.0f, "base"))
            val fast = readWav(synth(tts, 2.0f, 1.0f, "fast"))
            val high = readWav(synth(tts, 1.0f, 1.5f, "high"))
            val both = readWav(synth(tts, 2.0f, 1.5f, "both"))

            val f0Base = medianF0(base)
            val f0Fast = medianF0(fast)
            val f0High = medianF0(high)
            val f0Both = medianF0(both)
            Log.i(
                TAG,
                "dur base=${base.size} fast=${fast.size} high=${high.size} both=${both.size}",
            )
            Log.i(TAG, "f0 base=$f0Base fast=$f0Fast high=$f0High both=$f0Both")

            // Rate 2 halves the duration, at the same measured pitch.
            val rateRatio = base.size.toDouble() / fast.size
            assertTrue("rate duration ratio $rateRatio ~2", abs(rateRatio - 2.0) < 0.35)
            assertTrue("rate keeps pitch ($f0Base vs $f0Fast)", relDiff(f0Base, f0Fast) < 0.2)

            // Pitch 1.5 raises the fundamental ~1.5x, at the same duration.
            val pitchRatio = high.size.toDouble() / base.size
            assertTrue("pitch duration ratio $pitchRatio ~1", abs(pitchRatio - 1.0) < 0.2)
            val f0Ratio = f0High / f0Base
            assertTrue("pitch f0 ratio $f0Ratio ~1.5", abs(f0Ratio - 1.5) < 0.4)

            // Both compose: duration follows rate, pitch follows pitch.
            val bothDur = both.size.toDouble() / base.size
            assertTrue("both duration $bothDur ~0.5", abs(bothDur - 0.5) < 0.2)
            assertTrue("both keeps high pitch", relDiff(f0High, f0Both) < 0.2)
        } finally {
            tts.shutdown()
        }
    }

    // ---- WAV / DSP helpers ------------------------------------------------

    private fun readWav(file: File): FloatArray {
        val b = file.readBytes()
        require(b.size > 44 && b[0] == 'R'.code.toByte() && b[1] == 'I'.code.toByte())
        var p = 12
        var dataOff = -1
        var dataLen = 0
        while (p + 8 <= b.size) {
            val id = String(b, p, 4, Charsets.US_ASCII)
            val len = le32(b, p + 4)
            if (id == "data") { dataOff = p + 8; dataLen = len; break }
            p += 8 + len + (len and 1)
        }
        require(dataOff >= 0) { "no data chunk" }
        val n = dataLen / 2
        return FloatArray(n) {
            val lo = b[dataOff + it * 2].toInt() and 0xFF
            val hi = b[dataOff + it * 2 + 1].toInt()
            ((hi shl 8) or lo).toShort() / 32768f
        }
    }

    private fun le32(b: ByteArray, o: Int): Int =
        (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8) or
            ((b[o + 2].toInt() and 0xFF) shl 16) or ((b[o + 3].toInt() and 0xFF) shl 24)

    private fun relDiff(a: Double, b: Double): Double =
        if (a <= 0 || b <= 0) 1.0 else abs(a - b) / ((a + b) / 2)

    /** Median fundamental over voiced 1024-sample frames, by normalized autocorrelation. */
    private fun medianF0(x: FloatArray): Double {
        val sr = SAMPLE_RATE
        val frame = 1024
        val hop = 512
        val minLag = sr / 400
        val maxLag = sr / 70
        val f0s = ArrayList<Double>()
        var i = 0
        while (i + frame <= x.size) {
            var energy = 0.0
            for (j in 0 until frame) energy += x[i + j].toDouble() * x[i + j]
            val rms = sqrt(energy / frame)
            if (rms > 0.02) {
                var bestLag = 0
                var best = 0.0
                for (lag in minLag..maxLag) {
                    var dot = 0.0
                    var n = 0
                    for (j in 0 until frame - lag) { dot += x[i + j] * x[i + j + lag]; n++ }
                    val norm = dot / n
                    if (norm > best) { best = norm; bestLag = lag }
                }
                if (bestLag > 0) f0s.add(sr.toDouble() / bestLag)
            }
            i += hop
        }
        if (f0s.isEmpty()) return 0.0
        f0s.sort()
        return f0s[f0s.size / 2]
    }

    private companion object {
        const val TAG = "TtsVerify"
        const val ENGINE = "com.pockettts"
        const val SAMPLE_RATE = 24000
        const val TEXT = "The quick brown fox jumps over the lazy dog, and then it rests."
        const val SPEAK_TEXT =
            "Hello! I am Pocket TTS, a tiny hundred million parameter model speaking to you " +
                "from this phone. The quick brown fox jumps over the lazy dog, and then it " +
                "rests for a while before it runs again."
    }
}
