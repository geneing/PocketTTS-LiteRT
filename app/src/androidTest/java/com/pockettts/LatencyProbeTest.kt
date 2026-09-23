package com.pockettts

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.pockettts.Placement
import dev.pockettts.PocketTtsConfig
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import dev.pockettts.sonicStretch
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/**
 * Device harness for streaming: logs time-to-first-audio vs sentence length and
 * rate, and asserts the streamed audio still matches the one-shot take.
 */
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
            for (rate in listOf(1f, 1.4f, 1.5f)) {
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
            var diffSq = 0.0
            var sigSq = 0.0
            var dot = 0.0
            for (i in 0 until n) {
                val d = (oneShot[i] - streamed[i]).toDouble()
                maxD = maxOf(maxD, abs(oneShot[i] - streamed[i]))
                diffSq += d * d
                sigSq += oneShot[i].toDouble() * oneShot[i]
                dot += oneShot[i].toDouble() * streamed[i]
            }
            val rmsDiff = kotlin.math.sqrt(diffSq / n)
            val rmsSig = kotlin.math.sqrt(sigSq / n)
            val relDb = 20 * kotlin.math.log10(rmsDiff / rmsSig)
            Log.i(
                "Probe",
                "parity $label: oneShot=${oneShot.size} streamed=${streamed.size} max|d|=$maxD " +
                    "rmsDiff=$rmsDiff rmsSig=$rmsSig relDb=$relDb",
            )
            assertEquals("parity $label length", oneShot.size, streamed.size)
            assertTrue("parity $label relDb $relDb", relDb < -40.0)
        }
        // The streaming Sonic path should match shaping the one-shot take.
        for ((label, text) in texts) {
            val oneShot = engine.newSession("alba").use { it.synthesize(text).audio }
            val shaped = sonicStretch(oneShot, 1.5f)
            val session = engine.newSession("alba")
            session.rate = 1.5f
            val streamed = session.use { it.stream(text) {}.audio }
            val n = minOf(shaped.size, streamed.size)
            var maxD = 0f
            for (i in 0 until n) maxD = maxOf(maxD, abs(shaped[i] - streamed[i]))
            Log.i(
                "Probe",
                "rate1.5 $label: oneShotShaped=${shaped.size} streamed=${streamed.size} max|d|=$maxD",
            )
        }
        engine.close()
    }

    /**
     * Codec continuity: with [PocketTtsConfig.codecContinuity] the decoder is
     * primed from the previous sentence's tail. The primer is context, not
     * output, so the amount of audio must not change; and a single-sentence text
     * has nothing to prime from, so it must still match the one-shot take.
     */
    @Test
    fun continuity() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val models = PocketTtsModels.default(ctx)
        val dir = ctx.getExternalFilesDir(null) ?: ctx.filesDir
        val engine = PocketTtsEngine(
            ctx,
            PocketTtsConfig(
                models, Placement.default(ctx, dir), noiseSeed = 42L, codecContinuity = true,
            ),
        )
        for ((label, text) in texts) {
            val streamed = engine.newSession("alba").use { it.stream(text) {}.audio }
            val oneShot = engine.newSession("alba").use { it.synthesize(text).audio }
            val db = relDb(oneShot, streamed)
            Log.i(
                "Probe",
                "continuity $label: oneShot=${oneShot.size} streamed=${streamed.size} relDb=$db",
            )
            assertEquals("continuity $label length", oneShot.size, streamed.size)
            assertTrue("continuity $label finite", streamed.all { it.isFinite() })
            if (label == "short") {
                // One text chunk, so nothing was primed: full parity is expected.
                assertTrue("continuity $label relDb $db", db < -40.0)
            }
        }
        engine.close()
    }

    /**
     * Diagnostic for the two prompt-prefill paths. Batching the prompt through
     * the fused graph's `prefill` signature is *supposed* to reproduce the
     * per-token fused step, but on the shipped int8 graph it does not: with the
     * same text and seed the two takes are uncorrelated and the batched one
     * truncates short prompts (a six-word sentence collapses to three frames).
     * That is why [PocketTtsConfig.usePrefill] is off. This logs the gap and
     * asserts only what does hold: the per-token path is bit-reproducible,
     * because every utterance starts from a full KV reset.
     */
    @Test
    fun prefillPaths() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val models = PocketTtsModels.default(ctx)
        val dir = ctx.getExternalFilesDir(null) ?: ctx.filesDir
        val cases = listOf("short" to texts[0].second, "medium" to texts[1].second)
        val takes = HashMap<String, FloatArray>()
        for ((label, prefill) in listOf("batched" to true, "perToken" to false)) {
            // One engine at a time: each holds ~1 GB of native weights.
            val engine = PocketTtsEngine(
                ctx,
                PocketTtsConfig(
                    models, Placement.default(ctx, dir), noiseSeed = 42L, usePrefill = prefill,
                ),
            )
            for ((name, text) in cases) {
                val a = engine.newSession("alba").use { it.stream(text) {} }
                val b = engine.newSession("alba").use { it.stream(text) {} }
                Log.i(
                    "Probe",
                    "prefillPaths $label $name frames=${a.frames}/${b.frames} " +
                        "samples=${a.audio.size}/${b.audio.size} " +
                        "sec=${a.audio.size / 24000f} repeatRelDb=${relDb(a.audio, b.audio)}",
                )
                if (!prefill) {
                    assertArrayEquals("perToken $name repeat", a.audio, b.audio, 0f)
                }
                takes["$label/$name"] = b.audio
            }
            engine.close()
        }
        for ((name, _) in cases) {
            val batched = takes.getValue("batched/$name")
            val perToken = takes.getValue("perToken/$name")
            Log.i(
                "Probe",
                "prefillPaths $name batched-vs-perToken samples=${batched.size}/" +
                    "${perToken.size} relDb=${relDb(batched, perToken)}",
            )
        }
    }

    private fun relDb(a: FloatArray, b: FloatArray): Double {
        val n = minOf(a.size, b.size)
        var diffSq = 0.0
        var sigSq = 0.0
        for (i in 0 until n) {
            val d = (a[i] - b[i]).toDouble()
            diffSq += d * d
            sigSq += a[i].toDouble() * a[i]
        }
        return 20 * kotlin.math.log10(kotlin.math.sqrt(diffSq / n) / kotlin.math.sqrt(sigSq / n))
    }
}
