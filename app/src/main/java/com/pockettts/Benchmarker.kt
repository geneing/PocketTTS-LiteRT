package com.pockettts

import android.content.Context
import dev.pockettts.Accel
import dev.pockettts.Placement
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsConfig
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import dev.pockettts.TtsResult
import dev.pockettts.Wav
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
    private val models = PocketTtsModels.default(context)
    private val modelDir: File get() = context.getExternalFilesDir(null) ?: context.filesDir

    /** One benchmark scenario: a placement, frames per invocation, and the LM graph. */
    private data class Cfg(
        val p: Placement,
        val name: String,
        val lmSteps: Int = 1,
        val lmGraph: String? = null,
    )

    private companion object {
        const val I8_FP16 = "pt_flowlm_fused_fp16.tflite"
        const val I8_DYN8_ALL = "pt_flowlm_fused_dyn8_all.tflite"
        const val I8_DYN8_BODY = "pt_flowlm_fused_dyn8_body.tflite"
        const val I8_DYN4_ALL = "pt_flowlm_fused_dyn4_all.tflite"
        const val I8_ST8_BODY = "pt_flowlm_fused_st8_body.tflite"
        const val I8_WO8_ALL = "pt_flowlm_fused_wo8_all.tflite"
    }

    private fun engineFor(
        p: Placement,
        lmGraph: String? = null,
        lmSteps: Int = 1,
        streamW: Int = PocketTts.STREAM_W,
        gpuCache: File? = null,
    ) = PocketTtsEngine(
        context,
        PocketTtsConfig(
            models = models,
            placement = p,
            lmGraph = lmGraph,
            lmSteps = lmSteps,
            streamW = streamW,
            noiseSeed = seed,
            gpuCache = gpuCache,
        ),
    )

    fun run(text: String, voice: String, repeats: Int): String {
        val sb = StringBuilder()
        // gold first (it is the audio reference), then the NPU cases: LiteRT's
        // environment is process-global and the *first* load fixes dispatch
        // options, so an NPU load must not come after a CPU/GPU one.
        val placements = listOf(
            Cfg(Placement.GOLD, "gold_cpu", lmGraph = I8_FP16),
            Cfg(Placement(Accel.CPU, Accel.NPU, Accel.GPU), "fp16_dectx_npu", lmGraph = I8_FP16),
            Cfg(Placement(Accel.CPU, Accel.NPU, Accel.GPU), "i8_dyn8_all", lmGraph = I8_DYN8_ALL),
            Cfg(Placement(Accel.CPU, Accel.NPU, Accel.GPU), "i8_dyn8_body", lmGraph = I8_DYN8_BODY),
            Cfg(Placement(Accel.CPU, Accel.NPU, Accel.GPU), "i8_dyn4_all", lmGraph = I8_DYN4_ALL),
            Cfg(Placement(Accel.CPU, Accel.NPU, Accel.GPU), "i8_st8_body", lmGraph = I8_ST8_BODY),
            Cfg(Placement(Accel.CPU, Accel.NPU, Accel.GPU), "i8_wo8_all", lmGraph = I8_WO8_ALL),
            Cfg(Placement(Accel.CPU, Accel.CPU, Accel.CPU), "i8_dyn8_all_allcpu", lmGraph = I8_DYN8_ALL),
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
        val audioByName = HashMap<String, FloatArray>()

        sb.appendLine("== placements ==")
        for (cfg in placements) {
            val p = cfg.p
            val name = cfg.name
            val tLoad = System.nanoTime()
            val s = try {
                engineFor(p, cfg.lmGraph, cfg.lmSteps)
            } catch (e: Throwable) {
                sb.appendLine("[$name] ${p.label}: LOAD FAILED: ${e.message}")
                continue
            }
            val loadTotal = (System.nanoTime() - tLoad) / 1_000_000
            try {
                sb.appendLine()
                sb.appendLine("[$name] ${s.placements} | ${s.lmGraphName}")
                sb.appendLine(
                    "  load: lm ${s.loadMs["lm"]} dectx ${s.loadMs["dectx"]} dec ${s.loadMs["dec"]} ms " +
                        "(graph sum ${s.loadMs.values.sum()} ms, incl. assets/ctx $loadTotal ms)",
                )
                val rtf = ArrayList<Double>()
                var last: TtsResult? = null
                for (r in 1..repeats) {
                    val res = s.synthesize(text, voice)
                    last = res
                    val secs = res.audio.size.toDouble() / PocketTts.SAMPLE_RATE
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
                    audioByName[name] = audio
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

        // ---- streaming -------------------------------------------------------
        sb.appendLine()
        sb.appendLine("== streaming: audio chunks emitted while the LM is still generating ==")
        for (w in listOf(512, 1024, 2048)) {
            streamRow(sb, w, Accel.GPU, audioByName["i8_dyn8_all"], text, voice, repeats)
        }

        sb.appendLine()
        sb.appendLine("== load-time scenarios (two passes each; pass2 ~ warm page cache) ==")
        val scenarios = listOf(
            "default(policy)" to Placement.default(context, modelDir),
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
                engineFor(p, gpuCache = cache)
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

    private fun streamRow(
        sb: StringBuilder,
        w: Int,
        dec: Accel,
        ref: FloatArray?,
        text: String,
        voice: String,
        repeats: Int,
    ) {
        val graph = PocketTts.deconlyGraph(w)
        if (!File(modelDir, graph).exists()) {
            sb.appendLine("  w=$w dec=${dec.tag}: $graph not installed, skipped")
            return
        }
        val s = try {
            engineFor(Placement(Accel.CPU, Accel.NPU, dec), I8_DYN8_ALL, streamW = w)
        } catch (e: Throwable) {
            sb.appendLine("  w=$w dec=${dec.tag}: LOAD FAILED: ${e.message}")
            return
        }
        try {
            for (r in 1..repeats) {
                var chunks = 0
                val res = s.stream(text, voice) { chunks++ }
                val secs = res.audio.size.toDouble() / PocketTts.SAMPLE_RATE
                val pr = res.profile
                sb.appendLine(
                    "  w=$w dec=${dec.tag} run$r: ${f("%.2f", secs)}s audio, ${res.frames} frames, " +
                        "${res.ms} ms, ${f("%.2f", secs * 1000.0 / res.ms)}x RTF | " +
                        "first audio ${pr.firstChunkMs} ms | ${pr.audioChunks} chunks | " +
                        "lm ${pr.lmRunMs} dec_tx ${pr.decTxMs} seanet ${pr.seanetMs} ms",
                )
                if (r == 1) {
                    Wav.write(File(context.filesDir, "bench_stream_w${w}_${dec.tag}.wav"), res.audio)
                    if (ref != null) {
                        sb.appendLine(
                            "        vs one-shot: corr " +
                                "${f("%.6f", AudioQuality.compare(ref, res.audio).corr)} | " +
                                "max|d| ${f("%.3e", maxAbsDiff(ref, res.audio))} | " +
                                "len ${ref.size} vs ${res.audio.size}",
                        )
                    }
                }
            }
        } catch (e: Throwable) {
            sb.appendLine("  w=$w dec=${dec.tag} RUN FAILED: $e")
        } finally {
            s.close()
        }
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Double {
        val n = minOf(a.size, b.size)
        var m = 0.0
        for (i in 0 until n) m = maxOf(m, kotlin.math.abs(a[i] - b[i]).toDouble())
        return m
    }

    private fun median(v: List<Double>): Double {
        if (v.isEmpty()) return Double.NaN
        val s = v.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2.0
    }

    private fun f(fmt: String, vararg args: Any): String =
        String.format(Locale.US, fmt, *args)
}
