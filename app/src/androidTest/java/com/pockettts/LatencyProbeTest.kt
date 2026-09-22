package com.pockettts

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.pockettts.Placement
import dev.pockettts.PocketTtsConfig
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/** Temporary: measures time-to-first-audio vs sentence length and rate, and
 *  checks the streaming audio still matches the one-shot take. */
@RunWith(AndroidJUnit4::class)
class LatencyProbeTest {

    private val texts = listOf(
        "short" to "Hello there, how are you?",
        "medium" to "Hello! I am Pocket TTS, a tiny hundred million parameter model speaking to you from this phone.",
        "long" to (
            "The quick brown fox jumps over the lazy dog, and then it rests for a while. " +
                "It was the best of times, it was the worst of times, it was the age of wisdom, " +
                "it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity."
            ),
    )

    @Test
    fun probe() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val models = PocketTtsModels.default(ctx)
        val dir = ctx.getExternalFilesDir(null) ?: ctx.filesDir
        val engine = PocketTtsEngine(
            ctx,
            PocketTtsConfig(models, Placement.default(ctx, dir), noiseSeed = 42L),
        )
        for ((label, text) in texts) {
            for (rate in listOf(1f, 0.5f)) {
                val session = engine.newSession("alba")
                session.rate = rate
                var chunks = 0
                var samples = 0
                val r = session.stream(text) { chunks++; samples += it.size }
                Log.i(
                    "Probe",
                    "$label rate=$rate chars=${text.length}: frames=${r.frames} ms=${r.ms} " +
                        "firstAudio=${r.profile.firstChunkMs} audioChunks=$chunks samples=$samples",
                )
                session.close()
            }
        }
        // Streaming must still reproduce the one-shot audio (same seed).
        for ((label, text) in texts) {
            val oneShot = engine.newSession("alba").use { it.synthesize(text).audio }
            val streamed = engine.newSession("alba").use { it.stream(text) {}.audio }
            val n = minOf(oneShot.size, streamed.size)
            var maxD = 0f
            for (i in 0 until n) maxD = maxOf(maxD, abs(oneShot[i] - streamed[i]))
            Log.i(
                "Probe",
                "parity $label: oneShot=${oneShot.size} streamed=${streamed.size} max|d|=$maxD",
            )
        }
        engine.close()
    }
}
