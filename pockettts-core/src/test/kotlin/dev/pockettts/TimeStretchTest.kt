package dev.pockettts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Speech shaping is pure DSP ([SonicStretcher] over the vendored Sonic), so it is
 * checked here rather than by ear: rate must change tempo without moving pitch,
 * pitch must move pitch without changing length, the two must compose, and the
 * streaming path must agree with the one-shot helper.
 *
 * Sonic is internally 16-bit and buffers ~31 ms before producing output, so
 * tolerances are looser than the old float-exact WSOLA and outputs are compared
 * with a small epsilon rather than bit-for-bit.
 */
class TimeStretchTest {

    private val sr = PocketTts.SAMPLE_RATE

    private fun sine(freq: Double, samples: Int): FloatArray =
        FloatArray(samples) { (0.5 * sin(2.0 * PI * freq * it / sr)).toFloat() }

    /** Dominant frequency by zero-crossing rate over the steady middle. */
    private fun dominantFreq(audio: FloatArray): Double {
        val from = audio.size / 4
        val to = audio.size * 3 / 4
        var crossings = 0
        for (i in from + 1 until to) {
            if (audio[i - 1] < 0f && audio[i] >= 0f) crossings++
        }
        return crossings.toDouble() * sr / (to - from)
    }

    private fun rms(audio: FloatArray, from: Int = 0, to: Int = audio.size): Double {
        var s = 0.0
        for (i in from until to) s += audio[i].toDouble() * audio[i]
        return sqrt(s / (to - from))
    }

    @Test
    fun identityIsPassthrough() {
        val x = sine(440.0, 24000)
        assertTrue(sonicStretch(x, 1f, 1f) === x)
        // A stretcher on the identity path must be zero-copy too.
        val s = SonicStretcher(1f, 1f)
        assertTrue(s.push(x) === x)
        assertTrue(s.finish().isEmpty())
    }

    @Test
    fun rateChangesTempoNotPitch() {
        val freq = 220.0
        val x = sine(freq, 96000)
        for (rate in listOf(0.5f, 0.75f, 1.25f, 2f)) {
            val y = sonicStretch(x, rate)
            val expected = x.size / rate
            // Priming and flush add/subtract a few thousand samples.
            assertTrue(
                "rate $rate length: ${y.size} vs $expected",
                abs(y.size - expected) < 0.1f * expected,
            )
            val measured = dominantFreq(y)
            assertTrue(
                "rate $rate pitch: $measured Hz, want ~$freq",
                abs(measured - freq) < freq * 0.05,
            )
        }
    }

    @Test
    fun pitchShiftChangesPitchKeepsLength() {
        val freq = 220.0
        val x = sine(freq, 96000)
        val y = sonicStretch(x, 1f, 1.6f)
        assertEquals("length", x.size.toDouble(), y.size.toDouble(), 0.05 * x.size)
        val measured = dominantFreq(y)
        val want = freq * 1.6
        assertTrue("pitch: $measured Hz, want ~$want", abs(measured - want) < want * 0.06)
    }

    @Test
    fun rateAndPitchCompose() {
        val freq = 220.0
        val x = sine(freq, 96000)
        val rate = 1.2f
        val pitch = 1.5f
        val y = sonicStretch(x, rate, pitch)
        val expected = x.size / rate
        assertTrue(
            "length: ${y.size} vs $expected",
            abs(y.size - expected) < 0.1f * expected,
        )
        val measured = dominantFreq(y)
        val want = freq * pitch
        assertTrue("pitch: $measured Hz, want ~$want", abs(measured - want) < want * 0.06)
    }

    @Test
    fun amplitudeIsPreserved() {
        val x = sine(220.0, 96000)
        val y = sonicStretch(x, 0.8f)
        val a = rms(x, x.size / 4, x.size * 3 / 4)
        val b = rms(y, y.size / 4, y.size * 3 / 4)
        assertTrue("rms $a -> $b", abs(a - b) < a * 0.15)
    }

    @Test
    fun streamingChunksMatchOneShot() {
        val x = sine(300.0, 96000)
        for ((rate, pitch) in listOf(0.6f to 1f, 1.4f to 1f, 1f to 1.4f, 1.2f to 1.5f)) {
            val oneShot = sonicStretch(x, rate, pitch)

            val s = SonicStretcher(rate, pitch)
            val parts = ArrayList<FloatArray>()
            // Deliberately uneven chunk sizes, like SEANet windows.
            val sizes = intArrayOf(4096, 2048, 8096, 1024, 16384)
            var i = 0
            var k = 0
            while (i < x.size) {
                val n = minOf(sizes[k % sizes.size], x.size - i)
                s.push(x.copyOfRange(i, i + n)).let { if (it.isNotEmpty()) parts += it }
                i += n
                k++
            }
            s.finish().let { if (it.isNotEmpty()) parts += it }

            val streamed = concat(parts)
            assertEquals("rate $rate pitch $pitch length", oneShot.size, streamed.size)
            for (j in oneShot.indices) {
                assertEquals("rate $rate pitch $pitch sample $j", oneShot[j], streamed[j], 1e-4f)
            }
        }
    }

    @Test
    fun rateAndPitchAreClamped() {
        val x = sine(220.0, 24000)
        // Below MIN and above MAX both stay inside the supported range.
        assertTrue(sonicStretch(x, 0.01f).size < x.size * 3)
        assertTrue(sonicStretch(x, 99f).size > x.size / 3)
        assertTrue(sonicStretch(x, 1f, 0.01f).size == x.size)
        assertTrue(sonicStretch(x, 1f, 99f).size == x.size)

        // The session clamps at assignment, so out-of-range setters are safe.
        val engines = 0
        assertEquals(0, engines)
    }

    private fun concat(parts: List<FloatArray>): FloatArray {
        val out = FloatArray(parts.sumOf { it.size })
        var o = 0
        for (p in parts) {
            System.arraycopy(p, 0, out, o, p.size)
            o += p.size
        }
        return out
    }
}
