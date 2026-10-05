package com.pockettts

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.BatteryManager
import android.os.Build
import android.os.OutcomeReceiver
import android.os.SystemClock
import android.os.health.SystemHealthManager
import android.os.PowerMonitor
import android.os.PowerMonitorReadings
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.pockettts.Accel
import dev.pockettts.Placement
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsConfig
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import dev.pockettts.TtsResult
import dev.pockettts.Wav
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Repeatable Pixel power probe: the same long paragraph is played once from a
 * cached waveform (audio-only baseline), then synthesized while AudioTrack is
 * playing. PowerMonitor deltas include ODPM rails and modeled CPU/GPU/TPU
 * consumers, so the two intervals can be compared even while USB is attached.
 */
@RunWith(AndroidJUnit4::class)
class PowerBenchmarkTest {

    @Test
    fun longParagraphPlaybackPower() = runPowerBenchmark(cpuInt8Seanet = false)

    @Test
    fun longParagraphCpuInt8SeanetPower() = runPowerBenchmark(cpuInt8Seanet = true)

    @Test
    fun longParagraphFlowLmGpuFp16Power() = runFlowLmPowerBenchmark(
        caseName = "gpu-requested-fp16-step",
        lmPlacement = Accel.GPU,
        lmGraph = FP16_FLOWLM_GRAPH,
        lmSteps = 1,
    )

    @Test
    fun longParagraphFlowLmGpuFp16Ms4Power() = runFlowLmPowerBenchmark(
        caseName = "gpu-requested-fp16-ms4",
        lmPlacement = Accel.GPU,
        lmGraph = FP16_FLOWLM_GRAPH,
        lmSteps = 4,
    )

    @Test
    fun longParagraphFlowLmGpuDyn8Int8Power() = runFlowLmPowerBenchmark(
        caseName = "gpu-requested-dyn8-int8-step",
        lmPlacement = Accel.GPU,
        lmGraph = PocketTts.LM,
        lmSteps = 1,
    )

    private fun runPowerBenchmark(cpuInt8Seanet: Boolean) {
        assumeTrue("PowerMonitor requires Android 15 / API 35", Build.VERSION.SDK_INT >= 35)

        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val health = requireNotNull(context.getSystemService(SystemHealthManager::class.java))
        val monitors = supportedMonitors(health)
        assumeTrue("device does not expose power monitors", monitors.isNotEmpty())
        val relevant = monitors.filter { it.name.isRelevantPowerDomain() }
        assertTrue("no CPU/GPU/TPU energy monitors: ${monitors.map { it.name }}", relevant.isNotEmpty())

        val battery = requireNotNull(context.getSystemService(BatteryManager::class.java))
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        val defaultPlacement = Placement.default(context, dir)
        val referenceEngine = PocketTtsEngine(
            context,
            PocketTtsConfig(
                models = PocketTtsModels.default(context),
                placement = defaultPlacement,
                noiseSeed = 42L,
            ),
        )

        var referenceEngineClosed = false
        try {
            // Compile/initialize the selected delegates before taking any readings.
            val warmup = referenceEngine.synthesize("A short warmup sentence.", "alba")
            assertTrue("warmup produced no audio", warmup.audio.isNotEmpty())

            // Establish streaming-path RTF and the exact waveform used by the
            // audio-only control. Neither interval is included in power deltas.
            val reference = referenceEngine.stream(PARAGRAPH, "alba") {}
            assertTrue("reference produced no audio", reference.audio.isNotEmpty())

            val audioBaseline = measurePlayback(health, relevant, reference.audio)
            val referencePlayback = if (cpuInt8Seanet) {
                measureSynthesisPlayback(health, relevant, referenceEngine)
            } else {
                null
            }
            val referenceAudio = referencePlayback?.result?.audio ?: reference.audio
            assertEquals("reference replay audio length", reference.audio.size, referenceAudio.size)
            if (referencePlayback != null) {
                val referenceRepeatCorr = AudioQuality.compare(reference.audio, referenceAudio).corr
                assertTrue("reference repeat correlation: $referenceRepeatCorr", referenceRepeatCorr >= 0.99)
            }
            val sampleDir = context.getExternalFilesDir("power-benchmark")
                ?: File(context.filesDir, "power-benchmark")
            saveSample(sampleDir, "${if (cpuInt8Seanet) "seanet-cpu-int8" else "seanet-default"}-reference.wav", referenceAudio)

            val candidateEngine = if (cpuInt8Seanet) {
                referenceEngine.close()
                referenceEngineClosed = true
                val candidateGraph = File(dir, CPU_INT8_STREAM_GRAPH)
                assertTrue("missing CPU-int8 SEANet graph: ${candidateGraph.absolutePath}", candidateGraph.isFile)
                PocketTtsEngine(
                    context,
                    PocketTtsConfig(
                        models = PocketTtsModels.default(context),
                        placement = Placement(Accel.CPU, Accel.NPU, Accel.CPU),
                        streamW = 512,
                        streamDecoderGraph = CPU_INT8_STREAM_GRAPH,
                        noiseSeed = 42L,
                    ),
                )
            } else {
                referenceEngine
            }

            try {
                if (candidateEngine !== referenceEngine) {
                    val candidateWarmup = candidateEngine.stream("A short warmup sentence.", "alba") {}
                    assertTrue("CPU-int8 warmup produced no audio", candidateWarmup.audio.isNotEmpty())
                }
                val candidateSpeedRun = if (candidateEngine === referenceEngine) {
                    reference
                } else {
                    candidateEngine.stream(PARAGRAPH, "alba") {}
                }

                val fullPlayback = measureSynthesisPlayback(health, relevant, candidateEngine)
                saveSample(
                    sampleDir,
                    "${if (cpuInt8Seanet) "seanet-cpu-int8" else "seanet-default"}-candidate.wav",
                    fullPlayback.result.audio,
                )
                val quality = AudioQuality.compare(referenceAudio, fullPlayback.result.audio)

                val profile = fullPlayback.result.profile
                val timedModelMs = profile.lmRunMs + profile.decTxMs + profile.seanetMs
                val referenceRtf = reference.audio.size.toDouble() / PocketTts.SAMPLE_RATE /
                    (reference.ms / 1000.0)
                val candidateRtf = candidateSpeedRun.audio.size.toDouble() / PocketTts.SAMPLE_RATE /
                    (candidateSpeedRun.ms / 1000.0)
                val batteryLevel = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                val batteryCurrentUa = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)

                Log.i(
                    TAG,
                    "device=${Build.MODEL}/${Build.DEVICE} android=${Build.VERSION.SDK_INT} " +
                        "charging=${battery.isCharging} battery=${batteryLevel}% current=${batteryCurrentUa}uA " +
                        "screenInteractive=${context.getSystemService(android.os.PowerManager::class.java)?.isInteractive}",
                )
                Log.i(
                    TAG,
                    "placement=${candidateEngine.placements} decoderVariant=" +
                        "${if (cpuInt8Seanet) "cpu-int8" else "default"} " +
                        "monitors=${relevant.joinToString { it.name }}",
                )
                Log.i(
                    TAG,
                    "paragraph chars=${PARAGRAPH.length} words=${PARAGRAPH.split(Regex("\\s+")).size} " +
                        "audio=${reference.audio.size.toDouble() / PocketTts.SAMPLE_RATE}s " +
                        "frames=${fullPlayback.result.frames} referenceInference=${reference.ms}ms " +
                        "candidateInference=${candidateSpeedRun.ms}ms " +
                        "referenceRtf=${fmt(referenceRtf)}x candidateRtf=${fmt(candidateRtf)}x " +
                        "playbackWall=${fullPlayback.elapsedMs}ms " +
                        "qualityCorr=${fmt(quality.corr)}",
                )
                referencePlayback?.result?.profile?.let { profile ->
                    Log.i(
                        TAG,
                        "reference stage time: lm=${profile.lmRunMs}ms dectx=${profile.decTxMs}ms " +
                            "seanet=${profile.seanetMs}ms",
                    )
                }
                Log.i(
                    TAG,
                    "model stage time: lm=${profile.lmRunMs}ms dectx=${profile.decTxMs}ms " +
                        "seanet=${profile.seanetMs}ms total=${timedModelMs}ms " +
                        "shares=${fmt(percent(profile.lmRunMs, timedModelMs))}/" +
                        "${fmt(percent(profile.decTxMs, timedModelMs))}/" +
                        "${fmt(percent(profile.seanetMs, timedModelMs))}%",
                )
                logEnergy("audio-only", audioBaseline.deltaJoules)
                referencePlayback?.let {
                    logEnergy("reference synthesize+play", it.deltaJoules)
                    logEnergy(
                        "reference incremental model energy (full minus duration-scaled audio-only)",
                        incrementalEnergy(audioBaseline, it),
                    )
                }
                logEnergy("synthesize+play", fullPlayback.deltaJoules)
                logEnergy(
                    "incremental model energy (full minus duration-scaled audio-only)",
                    incrementalEnergy(audioBaseline, fullPlayback),
                )
                assertEquals("streamed audio length", reference.audio.size, fullPlayback.result.audio.size)
                assertTrue("playback run lost audio correlation: ${quality.corr}", quality.corr >= 0.99)
                if (cpuInt8Seanet) {
                    assertTrue(
                        "CPU-int8 streaming RTF fell below 95% of default: " +
                            "$candidateRtf vs $referenceRtf",
                        candidateRtf >= referenceRtf * 0.95,
                    )
                }
            } finally {
                if (candidateEngine !== referenceEngine) candidateEngine.close()
            }
        } finally {
            if (!referenceEngineClosed) referenceEngine.close()
        }
    }

    /** Paired CPU-reference / Flow-LM candidate energy and quality probe. */
    private fun runFlowLmPowerBenchmark(
        caseName: String,
        lmPlacement: Accel,
        lmGraph: String?,
        lmSteps: Int,
    ) {
        assumeTrue("PowerMonitor requires Android 15 / API 35", Build.VERSION.SDK_INT >= 35)

        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val health = requireNotNull(context.getSystemService(SystemHealthManager::class.java))
        val monitors = supportedMonitors(health)
        assumeTrue("device does not expose power monitors", monitors.isNotEmpty())
        val relevant = monitors.filter { it.name.isRelevantPowerDomain() }
        assertTrue("no CPU/GPU/TPU energy monitors: ${monitors.map { it.name }}", relevant.isNotEmpty())

        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        val models = PocketTtsModels.default(context)
        val defaultPlacement = Placement.default(context, dir)
        assumeTrue(
            "Flow-LM comparison requires default CPU placement; detected ${defaultPlacement.label}",
            defaultPlacement.lm == Accel.CPU,
        )

        val referenceGraph = lmGraph ?: PocketTts.LM
        assumeTrue("missing reference/candidate Flow-LM graph: $referenceGraph", models.store.exists(referenceGraph))
        if (lmSteps > 1) {
            val multiGraph = PocketTts.msGraph(lmSteps)
            assumeTrue("missing multistep Flow-LM graph: $multiGraph", models.store.exists(multiGraph))
        }

        val referenceEngine = PocketTtsEngine(
            context,
            PocketTtsConfig(
                models = models,
                placement = defaultPlacement,
                lmGraph = referenceGraph,
                lmSteps = 1,
                noiseSeed = FLOW_LM_SEED,
            ),
        )
        var referenceEngineClosed = false
        try {
            val warmup = referenceEngine.stream("A short warmup sentence.", "alba") {}
            assertTrue("reference warmup produced no audio", warmup.audio.isNotEmpty())

            val referenceTake = referenceEngine.stream(PARAGRAPH, "alba") {}
            assertTrue("reference produced no audio", referenceTake.audio.isNotEmpty())
            val audioBaseline = measurePlayback(health, relevant, referenceTake.audio)
            val referencePlayback = measureSynthesisPlayback(health, relevant, referenceEngine)
            val referenceRepeatQuality = AudioQuality.compare(referenceTake.audio, referencePlayback.result.audio)
            assertTrue(
                "reference repeat correlation: ${referenceRepeatQuality.corr}",
                referenceRepeatQuality.corr >= MIN_FLOW_LM_CORRELATION,
            )

            val sampleDir = context.getExternalFilesDir("power-benchmark")
                ?: File(context.filesDir, "power-benchmark")
            saveSample(sampleDir, "flowlm-$caseName-reference.wav", referencePlayback.result.audio)
            val referencePlacement = referenceEngine.placements
            referenceEngine.close()
            referenceEngineClosed = true

            val candidatePlacement = defaultPlacement.copy(lm = lmPlacement)
            val candidateEngine = PocketTtsEngine(
                context,
                PocketTtsConfig(
                    models = models,
                    placement = candidatePlacement,
                    lmGraph = referenceGraph,
                    lmSteps = lmSteps,
                    noiseSeed = FLOW_LM_SEED,
                ),
            )
            try {
                val candidateWarmup = candidateEngine.stream("A short warmup sentence.", "alba") {}
                assertTrue("candidate warmup produced no audio", candidateWarmup.audio.isNotEmpty())

                val candidateSpeedRun = candidateEngine.stream(PARAGRAPH, "alba") {}
                assertTrue("candidate speed run produced no audio", candidateSpeedRun.audio.isNotEmpty())

                val candidatePlayback = measureSynthesisPlayback(health, relevant, candidateEngine)
                saveSample(sampleDir, "flowlm-$caseName-candidate.wav", candidatePlayback.result.audio)

                val quality = AudioQuality.compare(referencePlayback.result.audio, candidatePlayback.result.audio)
                val referenceRtf = referenceTake.audio.size.toDouble() / PocketTts.SAMPLE_RATE /
                    (referenceTake.ms / 1000.0)
                val candidateRtf = candidateSpeedRun.audio.size.toDouble() / PocketTts.SAMPLE_RATE /
                    (candidateSpeedRun.ms / 1000.0)
                val audioSeconds = candidatePlayback.result.audio.size.toDouble() / PocketTts.SAMPLE_RATE
                val words = PARAGRAPH.split(Regex("\\s+")).size
                val candidateMultiGraph = if (lmSteps > 1) PocketTts.msGraph(lmSteps) else "none"
                val actualLmBackend = candidateEngine.runtimeAccelerators["lm"]
                val actualMultiBackend = candidateEngine.runtimeAccelerators["lm_ms"]

                Log.i(
                    TAG,
                    "Flow-LM paired case=$caseName seed=$FLOW_LM_SEED " +
                        "referencePlacement=$referencePlacement candidateRequestedPlacement=${candidateEngine.placements} " +
                        "candidateRuntimeAccelerators=${candidateEngine.runtimeAccelerators} " +
                        "referenceGraph=$referenceGraph candidateMultiGraph=$candidateMultiGraph",
                )
                Log.i(
                    TAG,
                    "paragraph chars=${PARAGRAPH.length} words=$words audio=${fmt(audioSeconds)}s " +
                        "referenceSamples=${referencePlayback.result.audio.size} " +
                        "candidateSamples=${candidatePlayback.result.audio.size} " +
                        "referenceNoPlay=${referenceTake.ms}ms candidateNoPlay=${candidateSpeedRun.ms}ms " +
                        "referenceRtf=${fmt(referenceRtf)}x candidateRtf=${fmt(candidateRtf)}x " +
                        "referencePlaybackWall=${referencePlayback.elapsedMs}ms " +
                        "candidatePlaybackWall=${candidatePlayback.elapsedMs}ms " +
                        "qualityCorr=${fmt(quality.corr)}",
                )
                logFlowLmStages("reference", referencePlayback.result, referencePlayback.elapsedMs)
                logFlowLmStages("candidate", candidatePlayback.result, candidatePlayback.elapsedMs)

                logEnergy("audio-only", audioBaseline.deltaJoules)
                logEnergy("reference synthesize+play", referencePlayback.deltaJoules)
                logEnergy(
                    "reference incremental model energy (full minus duration-scaled audio-only)",
                    incrementalEnergy(audioBaseline, referencePlayback),
                )
                logEnergy("candidate synthesize+play", candidatePlayback.deltaJoules)
                logEnergy(
                    "candidate incremental model energy (full minus duration-scaled audio-only)",
                    incrementalEnergy(audioBaseline, candidatePlayback),
                )

                assertTrue(
                    "Flow-LM $caseName correlation ${quality.corr} is below $MIN_FLOW_LM_CORRELATION",
                    quality.corr >= MIN_FLOW_LM_CORRELATION,
                )
                assertEquals("candidate/reference streamed audio length", referenceTake.audio.size, candidatePlayback.result.audio.size)
                assertEquals("requested Flow-LM backend loaded", lmPlacement, actualLmBackend)
                if (lmSteps > 1) {
                    assertEquals("requested multi-step Flow-LM backend loaded", lmPlacement, actualMultiBackend)
                }
            } finally {
                candidateEngine.close()
            }
        } finally {
            if (!referenceEngineClosed) referenceEngine.close()
        }
    }

    private fun logFlowLmStages(label: String, result: TtsResult, playbackElapsedMs: Long) {
        val profile = result.profile
        val audioSeconds = result.audio.size.toDouble() / PocketTts.SAMPLE_RATE
        Log.i(
            TAG,
            "$label streamMs=${result.ms}ms playbackWall=${playbackElapsedMs}ms " +
                "audio=${fmt(audioSeconds)}s frames=${result.frames} " +
                "lmSteps=${profile.lmSteps} lmInvocations=${profile.lmInvocations}",
        )
        Log.i(
            TAG,
            "$label stage time: lmInput=${profile.lmInMs}ms lmRun=${profile.lmRunMs}ms " +
                "lmRead=${profile.lmReadMs}ms dectx=${profile.decTxMs}ms seanet=${profile.seanetMs}ms",
        )
    }

    private fun measurePlayback(
        health: SystemHealthManager,
        monitors: List<PowerMonitor>,
        audio: FloatArray,
    ): Measurement {
        val track = newTrack()
        return try {
            val before = powerSnapshot(health, monitors)
            val startMs = SystemClock.elapsedRealtime()
            track.play()
            writeAudio(track, audio)
            drain(track, audio.size)
            val elapsedMs = SystemClock.elapsedRealtime() - startMs
            val after = powerSnapshot(health, monitors)
            Measurement(before, after, elapsedMs, deltaJoules(before, after))
        } finally {
            track.release()
        }
    }

    private fun measureSynthesisPlayback(
        health: SystemHealthManager,
        monitors: List<PowerMonitor>,
        engine: PocketTtsEngine,
    ): MeasurementWithResult {
        val track = newTrack()
        return try {
            val before = powerSnapshot(health, monitors)
            val startMs = SystemClock.elapsedRealtime()
            track.play()
            val result = engine.newSession("alba").use { session ->
                session.stream(PARAGRAPH) { chunk -> writeAudio(track, chunk) }
            }
            drain(track, result.audio.size)
            val elapsedMs = SystemClock.elapsedRealtime() - startMs
            val after = powerSnapshot(health, monitors)
            MeasurementWithResult(
                Measurement(before, after, elapsedMs, deltaJoules(before, after)),
                result,
            )
        } finally {
            track.release()
        }
    }

    private fun newTrack(): AudioTrack {
        val format = AudioFormat.Builder()
            .setSampleRate(PocketTts.SAMPLE_RATE)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        val minBytes = AudioTrack.getMinBufferSize(
            PocketTts.SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        check(minBytes > 0) { "AudioTrack does not support 24 kHz mono float PCM" }
        return AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
            format,
            maxOf(minBytes, PocketTts.SAMPLE_RATE * Float.SIZE_BYTES),
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        ).apply { setVolume(0.5f) }
    }

    private fun writeAudio(track: AudioTrack, audio: FloatArray) {
        var offset = 0
        while (offset < audio.size) {
            val written = track.write(
                audio,
                offset,
                minOf(4096, audio.size - offset),
                AudioTrack.WRITE_BLOCKING,
            )
            check(written > 0) { "AudioTrack.write failed: $written" }
            offset += written
        }
    }

    private fun drain(track: AudioTrack, samples: Int) {
        val deadline = SystemClock.elapsedRealtime() + samples * 1000L / PocketTts.SAMPLE_RATE + 15_000L
        while (track.playbackHeadPosition < samples && SystemClock.elapsedRealtime() < deadline) {
            SystemClock.sleep(20)
        }
        assertTrue("AudioTrack did not finish playback", track.playbackHeadPosition >= samples)
        track.stop()
    }

    private fun supportedMonitors(manager: SystemHealthManager): List<PowerMonitor> {
        val latch = CountDownLatch(1)
        var result: List<PowerMonitor> = emptyList()
        manager.getSupportedPowerMonitors(null) {
            result = it.toList()
            latch.countDown()
        }
        assertTrue("timed out listing device power monitors", latch.await(30, TimeUnit.SECONDS))
        return result
    }

    private fun powerSnapshot(
        manager: SystemHealthManager,
        monitors: List<PowerMonitor>,
    ): Snapshot {
        val latch = CountDownLatch(1)
        var readings: PowerMonitorReadings? = null
        var failure: RuntimeException? = null
        manager.getPowerMonitorReadings(
            monitors,
            null,
            object : OutcomeReceiver<PowerMonitorReadings, RuntimeException> {
                override fun onResult(result: PowerMonitorReadings) {
                    readings = result
                    latch.countDown()
                }

                override fun onError(error: RuntimeException) {
                    failure = error
                    latch.countDown()
                }
            },
        )
        assertTrue("timed out reading device power monitors", latch.await(30, TimeUnit.SECONDS))
        failure?.let { throw AssertionError("could not read device power monitors", it) }
        val result = requireNotNull(readings)
        val values = monitors.associateWith { monitor -> result.getConsumedEnergy(monitor) }
        return Snapshot(values)
    }

    private fun deltaJoules(before: Snapshot, after: Snapshot): Map<String, Double> =
        before.values.mapNotNull { (monitor, start) ->
            val end = after.values[monitor] ?: return@mapNotNull null
            if (start < 0 || end < start) return@mapNotNull null
            monitor.name to ((end - start) / 1_000_000.0)
        }.toMap()

    private fun incrementalEnergy(baseline: Measurement, full: MeasurementWithResult): Map<String, Double> {
        val playbackScale = full.elapsedMs.toDouble() / baseline.elapsedMs.coerceAtLeast(1)
        return full.measurement.deltaJoules.mapValues { (name, joules) ->
            joules - (baseline.deltaJoules[name] ?: 0.0) * playbackScale
        }
    }

    private fun logEnergy(label: String, joules: Map<String, Double>) {
        val rendered = joules.entries
            .sortedBy { it.key }
            .joinToString { (name, value) -> "$name=${fmt(value)}J" }
        Log.i(TAG, "$label: $rendered")
    }

    private fun saveSample(directory: File, name: String, audio: FloatArray) {
        assertTrue("could not create sample directory: ${directory.absolutePath}", directory.mkdirs() || directory.isDirectory)
        val file = File(directory, name)
        Wav.write(file, audio)
        Log.i(TAG, "saved sample=${file.absolutePath} bytes=${file.length()}")
    }

    private fun String.isRelevantPowerDomain(): Boolean {
        val key = lowercase(Locale.ROOT)
        return key.contains("cpu") || key.contains("gpu") || key.contains("tpu") || key.contains("display")
    }

    private fun percent(part: Long, total: Long): Double =
        if (total > 0) part * 100.0 / total else 0.0

    private fun fmt(value: Double): String = String.format(Locale.US, "%.3f", value)

    private data class Snapshot(val values: Map<PowerMonitor, Long>)

    private data class Measurement(
        val before: Snapshot,
        val after: Snapshot,
        val elapsedMs: Long,
        val deltaJoules: Map<String, Double>,
    )

    private data class MeasurementWithResult(
        val measurement: Measurement,
        val result: TtsResult,
    ) {
        val elapsedMs: Long get() = measurement.elapsedMs
        val deltaJoules: Map<String, Double> get() = measurement.deltaJoules
    }

    private companion object {
        const val TAG = "PocketTTSPower"
        const val CPU_INT8_STREAM_GRAPH = "pt_mimi_deconly_w512_dyn8.tflite"
        const val FP16_FLOWLM_GRAPH = "pt_flowlm_fused_fp16.tflite"
        const val FLOW_LM_SEED = 42L
        const val MIN_FLOW_LM_CORRELATION = 0.99
        val PARAGRAPH = """
            Each spring, a small group of neighbors meets at the public library to plan a weekend repair fair. They bring lamps with loose switches, radios that have gone quiet, bicycles with stubborn brakes, and kitchen tools that only need a little attention. Before the doors open, volunteers arrange the tables by task and place a handwritten sign beside every box of spare parts. A retired engineer shows the children how to trace a simple circuit, while a local baker sets out warm bread and explains how patient practice can turn a difficult recipe into an ordinary part of the day.

            By midmorning, the room is busy but calm. People take turns describing what stopped working, and the volunteers ask questions before reaching for a screwdriver. Some repairs succeed quickly; others become lessons in what to try next. Nobody is asked to pay, and nobody is hurried toward a perfect result. The goal is to help useful things last longer, share skills that might otherwise remain hidden, and make it easier for strangers to begin a conversation. At the end of the afternoon, the tables are cleared, the tools are counted, and a list of unfinished jobs is saved for next month.
        """.trimIndent().replace('\n', ' ')
    }
}
