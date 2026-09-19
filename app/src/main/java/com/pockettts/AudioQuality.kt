package com.pockettts

import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Waveform comparison of a candidate decode against the CPU/CPU/CPU reference
 * ("gold"). Every run uses the same noise seed, so the audio should be
 * sample-aligned apart from delegate rounding; the metrics measure how far a
 * placement drifts from the reference.
 *
 *  * `corr`           peak normalized cross-correlation over ±maxLag samples
 *  * `snrDb`          10 log10(reference energy / residual energy), gain matched
 *  * `highBandErrDb`  the same on the first difference (a cheap high-pass), which
 *                     exposes the hissy/gravelly artifacts HNR is sensitive to
 *  * `hnrDb`          frame-wise harmonics-to-noise ratio of each signal alone
 */
object AudioQuality {

    data class Metrics(
        val refSamples: Int,
        val candSamples: Int,
        val lag: Int,
        val corr: Double,
        val snrDb: Double,
        val refHnrDb: Double,
        val candHnrDb: Double,
        val highBandErrDb: Double,
    )

    private const val SR = PocketTtsSynthesizer.SAMPLE_RATE

    fun compare(ref: FloatArray, cand: FloatArray, maxLag: Int = 128): Metrics {
        val al = align(ref, cand, maxLag)
        val r0 = max(0, al.lag)
        val c0 = max(0, -al.lag)
        val n = al.n
        val a = if (al.syy > 1e-12) al.sxy / al.syy else 1.0
        var err = 0.0
        var hbRef = 0.0
        var hbErr = 0.0
        var i = 1
        while (i < n) {
            val dr = ref[r0 + i].toDouble()
            val dc = a * cand[c0 + i].toDouble()
            val e = dr - dc
            err += e * e
            val ddr = dr - ref[r0 + i - 1]
            val ddc = dc - a * cand[c0 + i - 1].toDouble()
            hbRef += ddr * ddr
            val he = ddr - ddc
            hbErr += he * he
            i++
        }
        val snr = 10.0 * log10((al.sxx + 1e-12) / (err + 1e-12))
        val hb = 10.0 * log10((hbRef + 1e-12) / (hbErr + 1e-12))
        return Metrics(
            refSamples = ref.size,
            candSamples = cand.size,
            lag = al.lag,
            corr = al.corr,
            snrDb = snr,
            refHnrDb = hnr(ref),
            candHnrDb = hnr(cand),
            highBandErrDb = hb,
        )
    }

    private class Align(
        val lag: Int,
        val corr: Double,
        val sxx: Double,
        val syy: Double,
        val sxy: Double,
        val n: Int,
    )

    private fun align(ref: FloatArray, cand: FloatArray, maxLag: Int): Align {
        var bestCorr = -2.0
        var bl = 0
        var bx = 0.0
        var by = 0.0
        var bxy = 0.0
        var bn = 0
        for (lag in -maxLag..maxLag) {
            val r0 = max(0, lag)
            val c0 = max(0, -lag)
            val n = min(ref.size - r0, cand.size - c0)
            if (n < 2) continue
            var sxx = 0.0
            var syy = 0.0
            var sxy = 0.0
            var i = 0
            while (i < n) {
                val x = ref[r0 + i].toDouble()
                val y = cand[c0 + i].toDouble()
                sxx += x * x; syy += y * y; sxy += x * y
                i++
            }
            val c = sxy / (sqrt(sxx) * sqrt(syy) + 1e-12)
            if (c > bestCorr) {
                bestCorr = c; bl = lag; bx = sxx; by = syy; bxy = sxy; bn = n
            }
        }
        return Align(bl, bestCorr, bx, by, bxy, bn)
    }

    /**
     * Frame-wise harmonics-to-noise ratio (dB), averaged over voiced frames.
     * Per 40 ms frame: normalized autocorrelation peak over a 60–400 Hz pitch
     * range, HNR = 10 log10(r/(1-r)). Rough but comparable across placements.
     */
    fun hnr(x: FloatArray): Double {
        val frame = SR / 25                  // 40 ms
        val lagMin = SR / 400                // 400 Hz
        val lagMax = SR / 60                 // 60 Hz
        var maxE = 0.0
        var s = 0
        while (s + frame <= x.size) {
            var e = 0.0
            var i = 0
            while (i < frame) { val v = x[s + i].toDouble(); e += v * v; i++ }
            if (e > maxE) maxE = e
            s += frame
        }
        if (maxE <= 0.0) return Double.NaN
        var sum = 0.0
        var cnt = 0
        s = 0
        while (s + frame <= x.size) {
            var e = 0.0
            var i = 0
            while (i < frame) { val v = x[s + i].toDouble(); e += v * v; i++ }
            if (e > 1e-4 * maxE) {
                var best = 0.0
                val hi = min(lagMax, frame - 1)
                var lag = lagMin
                while (lag <= hi) {
                    var r = 0.0
                    var j = 0
                    while (j + lag < frame) { r += x[s + j].toDouble() * x[s + j + lag]; j++ }
                    r /= e
                    if (r > best) best = r
                    lag++
                }
                if (best > 0.05 && best < 0.999) {
                    sum += 10.0 * log10(best / (1.0 - best))
                    cnt++
                }
            }
            s += frame
        }
        return if (cnt > 0) sum / cnt else Double.NaN
    }
}
