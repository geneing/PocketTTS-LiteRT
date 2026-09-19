package com.pockettts

import android.content.Context
import java.io.File
import java.util.Locale

/**
 * On-device benchmark. Loads the model under several placements, repeats
 * synthesis with a fixed noise seed, and compares each candidate's audio
 * against the CPU/CPU/CPU reference. Writes a text report plus `bench_*.wav`
 * into filesDir (adb-pullable) and mirrors the report to logcat tag
 * `PocketTTSBench`.
 */
class Benchmarker(private val context: Context) {

    private val seed = 20260919L
    private val lmBenchSteps = 128

    fun run(text: String, voice: String, repeats: Int): String {
        val sb = StringBuilder()
        val placements = listOf(
            Placement.GOLD to "gold_cpu",
            Placement(Accel.CPU, Accel.CPU, Accel.GPU) to "lm_cpu_dec_gpu",
            Placement(Accel.CPU, Accel.GPU, Accel.GPU) to "lm_cpu_all_gpu",
            Placement(Accel.GPU, Accel.CPU, Accel.GPU) to "shipped_gpu_lm",
            Placement(Accel.GPU32, Accel.CPU, Accel.GPU) to "lm_gpu32",
            Placement(Accel.GPU, Accel.GPU, Accel.GPU) to "all_gpu",
        )

        sb.appendLine("Pocket TTS benchmark")
        sb.appendLine(
            "device   : ${android.os.Build.MODEL} (${android.os.Build.DEVICE}), " +
                "Android ${android.os.Build.VERSION.RELEASE}",
        )
        sb.appendLine("renderer : ${Placement.renderer()}")
        sb.appendLine("text     : ${text.length} chars, voice=$voice, repeats=$repeats, seed=$seed")
        sb.appendLine()

        var gold: FloatArray? = null

        sb.appendLine("== placements ==")
        for ((p, name) in placements) {
            val tLoad = System.nanoTime()
            val s = try {
                PocketTtsSynthesizer(context, p, seed)
            } catch (e: Throwable) {
                sb.appendLine("[$name] ${p.label}: LOAD FAILED: ${e.message}")
                continue
            }
            val loadTotal = (System.nanoTime() - tLoad) / 1_000_000
            try {
                sb.appendLine()
                sb.appendLine("[$name] ${p.label}")
                sb.appendLine(
                    "  load: lm ${s.loadMs["lm"]} dectx ${s.loadMs["dectx"]} dec ${s.loadMs["dec"]} ms " +
                        "(graph sum ${s.loadMs.values.sum()} ms, incl. assets/ctx $loadTotal ms)",
                )
                val rtf = ArrayList<Double>()
                var last: PocketTtsSynthesizer.Result? = null
                for (r in 1..repeats) {
                    val res = s.synthesize(text, voice)
                    last = res
                    val secs = res.audio.size.toDouble() / PocketTtsSynthesizer.SAMPLE_RATE
                    val x = secs * 1000.0 / res.ms
                    rtf.add(x)
                    val pr = res.profile
                    sb.appendLine(
                        "  run$r: ${f("%.2f", secs)}s audio, ${res.frames} frames, ${res.ms} ms, " +
                            "${f("%.2f", x)}x RTF | lm ${pr.lmSteps} steps " +
                            "(${pr.promptSteps} prompt + ${pr.genFrames} gen): " +
                            "in ${pr.lmInMs} run ${pr.lmRunMs} read ${pr.lmReadMs} ms | " +
                            "dec_tx ${pr.decTxMs} seanet ${pr.seanetMs} ms",
                    )
                }
                sb.appendLine(
                    "  RTF: median ${f("%.2f", median(rtf))}x " +
                        "(min ${f("%.2f", rtf.minOrNull() ?: Double.NaN)} " +
                        "max ${f("%.2f", rtf.maxOrNull() ?: Double.NaN)})",
                )
                val audio = last?.audio
                if (audio != null) {
                    if (p == Placement.GOLD) {
                        gold = audio
                        Wav.write(File(context.filesDir, "bench_gold.wav"), audio)
                    } else {
                        Wav.write(File(context.filesDir, "bench_$name.wav"), audio)
                    }
                }
                val g = gold
                if (p != Placement.GOLD && g != null && audio != null) {
                    val m = AudioQuality.compare(g, audio)
                    sb.appendLine(
                        "  vs gold: corr ${f("%.4f", m.corr)} (lag ${m.lag}) | " +
                            "SNR ${f("%.1f", m.snrDb)} dB | high-band err ${f("%.1f", m.highBandErrDb)} dB | " +
                            "HNR gold ${f("%.1f", m.refHnrDb)} / cand ${f("%.1f", m.candHnrDb)} dB | " +
                            "len gold ${m.refSamples} cand ${m.candSamples}",
                    )
                }
                val mb = s.microBenchLm(lmBenchSteps, voice)
                if (mb.lmSteps > 0) {
                    val st = mb.lmSteps.toDouble()
                    sb.appendLine(
                        "  lm micro: ${mb.lmSteps} frames / ${mb.lmInvocations} inv | " +
                            "in ${f("%.2f", mb.lmInMs / st)} run ${f("%.2f", mb.lmRunMs / st)} " +
                            "read ${f("%.2f", mb.lmReadMs / st)} ms/frame | " +
                            "${f("%.1f", mb.lmInBytes / 1e6 / st)} MB in + " +
                            "${f("%.2f", mb.lmOutBytes / 1e6 / st)} MB out per invocation",
                    )
                }
            } catch (e: Throwable) {
                sb.appendLine("  RUN FAILED: $e")
            } finally {
                s.close()
            }
        }

        sb.appendLine()
        sb.appendLine("== load-time scenarios (two passes each; pass2 ~ warm page cache) ==")
        val scenarios = listOf(
            "default(policy)" to Placement.default(context),
            "all-GPU" to Placement(Accel.GPU, Accel.GPU, Accel.GPU),
            "all-CPU" to Placement.GOLD,
        )
        for ((name, p) in scenarios) appendLoad(sb, name, p)

        sb.appendLine()
        sb.appendLine("== GPU program cache (all-GPU; pass2 should hit the serialized cache) ==")
        val cache = File(context.cacheDir, "litert_program_cache").apply { mkdirs() }
        repeat(2) { n -> appendLoad(sb, "all-GPU+cache", Placement(Accel.GPU, Accel.GPU, Accel.GPU), cache, n + 1) }

        val report = sb.toString()
        File(context.filesDir, "benchmark.txt").writeText(report)
        report.lines().forEach { android.util.Log.i("PocketTTSBench", it) }
        return report
    }

    private fun appendLoad(
        sb: StringBuilder,
        name: String,
        p: Placement,
        cache: File? = null,
        pass: Int = -1,
    ) {
        repeat(if (pass < 0) 2 else 1) { n ->
            val label = if (pass < 0) "$name pass${n + 1}" else "$name pass$pass"
            val t = System.nanoTime()
            val s = try {
                PocketTtsSynthesizer(context, p, gpuCache = cache)
            } catch (e: Throwable) {
                sb.appendLine("$label: FAILED: ${e.message}")
                null
            }
            if (s != null) {
                sb.appendLine(
                    "$label: lm ${s.loadMs["lm"]} dectx ${s.loadMs["dectx"]} dec ${s.loadMs["dec"]} ms " +
                        "(total ${(System.nanoTime() - t) / 1_000_000} ms)",
                )
                s.close()
            }
        }
    }

    private fun median(v: List<Double>): Double {
        if (v.isEmpty()) return Double.NaN
        val s = v.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2.0
    }

    private fun f(fmt: String, vararg args: Any): String =
        String.format(Locale.US, fmt, *args)
}
