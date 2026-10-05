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
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

/**
 * Repeatable Pixel power probe: the same long paragraph is played once from a
 * cached waveform (audio-only baseline), then synthesized while AudioTrack is
 * playing. PowerMonitor deltas include ODPM rails and modeled CPU/GPU/TPU
 * consumers, so the two intervals can be compared even while USB is attached.
 */
@RunWith(AndroidJUnit4::class)
class PowerBenchmarkTest {

    @Test
    fun longParagraphPlaybackPower() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelDir = context.getExternalFilesDir(null) ?: context.filesDir
        val placement = Placement.default(context, modelDir)
        runPowerProbe("default", placement, placement, null, placement.deconly)
    }

    @Test
    fun phasePackedSeanetNpuLongParagraphPlaybackPower() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelDir = context.getExternalFilesDir(null) ?: context.filesDir
        val placement = Placement.default(context, modelDir).copy(deconly = Accel.GPU)
        runPowerProbe("seanet-npu-phase-w512", placement, placement, PHASE_GRAPH, Accel.NPU)
    }

    @Test
    fun phasePackedSeanetNpu1024LongParagraphPlaybackPower() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelDir = context.getExternalFilesDir(null) ?: context.filesDir
        val placement = Placement.default(context, modelDir).copy(deconly = Accel.GPU)
        runPowerProbe(
            "seanet-npu-phase-w1024", placement, placement, PHASE_GRAPH_1024, Accel.NPU,
            candidateWindow = 1024,
        )
    }

    private fun runPowerProbe(
        caseName: String,
        referencePlacement: Placement,
        candidatePlacement: Placement,
        candidateGraph: String?,
        candidateStreamAccel: Accel,
        candidateWindow: Int = PocketTts.STREAM_W,
    ) {
        assumeTrue("PowerMonitor requires Android 15 / API 35", Build.VERSION.SDK_INT >= 35)

        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val health = requireNotNull(context.getSystemService(SystemHealthManager::class.java))
        val monitors = supportedMonitors(health)
        assumeTrue("device does not expose power monitors", monitors.isNotEmpty())
        val relevant = monitors.filter { it.name.isRelevantPowerDomain() }
        assertTrue("no CPU/GPU/TPU energy monitors: ${monitors.map { it.name }}", relevant.isNotEmpty())

        val battery = requireNotNull(context.getSystemService(BatteryManager::class.java))
        val models = PocketTtsModels.default(context)
        val referenceEngine = newEngine(context, models, referencePlacement, null, referencePlacement.deconly)
        val sameEngine = candidateGraph == null && candidatePlacement == referencePlacement &&
            candidateStreamAccel == referencePlacement.deconly
        val candidateEngine = if (sameEngine) referenceEngine else {
            newEngine(context, models, candidatePlacement, candidateGraph, candidateStreamAccel, candidateWindow)
        }

        try {
            val warmup = referenceEngine.stream("A short warmup sentence.", "alba") {}
            assertTrue("warmup produced no audio", warmup.audio.isNotEmpty())
            if (candidateEngine !== referenceEngine) {
                val candidateWarmup = candidateEngine.stream("A short warmup sentence.", "alba") {}
                assertTrue("candidate warmup produced no audio", candidateWarmup.audio.isNotEmpty())
            }

            val reference = referenceEngine.stream(PARAGRAPH, "alba") {}
            assertTrue("reference produced no audio", reference.audio.isNotEmpty())
            val candidateProbe = if (candidateEngine === referenceEngine) reference else {
                candidateEngine.stream(PARAGRAPH, "alba") {}
            }
            assertTrue("candidate probe produced no audio", candidateProbe.audio.isNotEmpty())
            val audioBaseline = measurePlayback(health, relevant, reference.audio)
            val fullPlayback = measureSynthesisPlayback(health, relevant, candidateEngine)
            val audioCorr = correlation(reference.audio, fullPlayback.result.audio)

            val durationSeconds = reference.audio.size.toDouble() / PocketTts.SAMPLE_RATE
            val referenceRtf = durationSeconds / (reference.ms / 1000.0)
            val candidateRtf = durationSeconds / (candidateProbe.ms / 1000.0)

            val profile = fullPlayback.result.profile
            val timedModelMs = profile.lmRunMs + profile.decTxMs + profile.seanetMs
            val batteryLevel = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            val batteryCurrentUa = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            val sampleDir = context.getExternalFilesDir("power-benchmark") ?: context.filesDir
            sampleDir.mkdirs()
            val referenceWav = writeWav(File(sampleDir, "$caseName-reference.wav"), reference.audio)
            val candidateWav = writeWav(File(sampleDir, "$caseName-candidate.wav"), fullPlayback.result.audio)

            assertEquals("reference/candidate probe length", reference.audio.size, candidateProbe.audio.size)
            assertEquals("streamed audio length", reference.audio.size, fullPlayback.result.audio.size)
            assertTrue("playback run lost audio correlation: $audioCorr", audioCorr >= 0.99)
            if (candidateGraph != null) {
                assertTrue(
                    "candidate RTF regressed by more than 10%: reference=${fmt(referenceRtf)}x " +
                        "candidate=${fmt(candidateRtf)}x",
                    candidateRtf >= referenceRtf * 0.90,
                )
            }

            Log.i(
                TAG,
                "device=${Build.MODEL}/${Build.DEVICE} android=${Build.VERSION.SDK_INT} " +
                    "charging=${battery.isCharging} battery=${batteryLevel}% current=${batteryCurrentUa}uA " +
                    "screenInteractive=${context.getSystemService(android.os.PowerManager::class.java)?.isInteractive}",
            )
            Log.i(TAG, "case=$caseName reference=${referenceEngine.placements} candidate=${candidateEngine.placements}")
            Log.i(
                TAG,
                "paragraph chars=${PARAGRAPH.length} words=${PARAGRAPH.split(Regex("\\s+")).size} " +
                    "audio=${durationSeconds}s frames=${fullPlayback.result.frames} " +
                    "referenceInference=${reference.ms}ms candidateInference=${candidateProbe.ms}ms " +
                    "referenceRtf=${fmt(referenceRtf)}x candidateRtf=${fmt(candidateRtf)}x " +
                    "playbackWall=${fullPlayback.elapsedMs}ms audioCorr=${fmt(audioCorr)}",
            )
            Log.i(
                TAG,
                "candidate stage time: lm=${profile.lmRunMs}ms dectx=${profile.decTxMs}ms " +
                    "seanet=${profile.seanetMs}ms total=${timedModelMs}ms " +
                    "shares=${fmt(percent(profile.lmRunMs, timedModelMs))}/" +
                    "${fmt(percent(profile.decTxMs, timedModelMs))}/" +
                    "${fmt(percent(profile.seanetMs, timedModelMs))}%",
            )
            Log.i(TAG, "listening samples: reference=$referenceWav candidate=$candidateWav")
            logEnergy("audio-only", audioBaseline.deltaJoules)
            logEnergy("synthesize+play", fullPlayback.deltaJoules)
            logEnergy(
                "incremental model energy (full minus duration-scaled audio-only)",
                incrementalEnergy(audioBaseline, fullPlayback),
            )
        } finally {
            if (candidateEngine !== referenceEngine) candidateEngine.close()
            referenceEngine.close()
        }
    }

    private fun newEngine(
        context: android.content.Context,
        models: PocketTtsModels,
        placement: Placement,
        graph: String?,
        streamAccel: Accel,
        streamWindow: Int = PocketTts.STREAM_W,
    ) = PocketTtsEngine(
        context,
        PocketTtsConfig(
            models = models,
            placement = placement,
            noiseSeed = 42L,
            streamW = streamWindow,
            streamDecoderGraph = graph,
            streamDecoderAccel = streamAccel,
        ),
    )

    private fun correlation(a: FloatArray, b: FloatArray): Double {
        val n = minOf(a.size, b.size)
        require(n > 1) { "not enough samples to compare" }
        var sumA = 0.0
        var sumB = 0.0
        for (i in 0 until n) { sumA += a[i]; sumB += b[i] }
        val meanA = sumA / n
        val meanB = sumB / n
        var cov = 0.0
        var varA = 0.0
        var varB = 0.0
        for (i in 0 until n) {
            val da = a[i] - meanA
            val db = b[i] - meanB
            cov += da * db
            varA += da * da
            varB += db * db
        }
        return cov / kotlin.math.sqrt(varA * varB)
    }

    private fun writeWav(file: File, audio: FloatArray): String {
        val pcm = ByteBuffer.allocate(audio.size * Short.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        audio.forEach { sample ->
            pcm.putShort((sample.coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt().toShort())
        }
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(36 + pcm.capacity())
        header.put("WAVEfmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)
        header.putShort(1)
        header.putShort(1)
        header.putInt(PocketTts.SAMPLE_RATE)
        header.putInt(PocketTts.SAMPLE_RATE * Short.SIZE_BYTES)
        header.putShort(Short.SIZE_BYTES.toShort())
        header.putShort(16)
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(pcm.capacity())
        file.outputStream().use { out ->
            out.write(header.array())
            out.write(pcm.array())
        }
        return file.absolutePath
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
        const val PHASE_GRAPH = "pt_mimi_deconly_w512_phase_fp16.tflite"
        const val PHASE_GRAPH_1024 = "pt_mimi_deconly_w1024_phase_fp16.tflite"
        val PARAGRAPH = """
            Each spring, a small group of neighbors meets at the public library to plan a weekend repair fair. They bring lamps with loose switches, radios that have gone quiet, bicycles with stubborn brakes, and kitchen tools that only need a little attention. Before the doors open, volunteers arrange the tables by task and place a handwritten sign beside every box of spare parts. A retired engineer shows the children how to trace a simple circuit, while a local baker sets out warm bread and explains how patient practice can turn a difficult recipe into an ordinary part of the day.

            By midmorning, the room is busy but calm. People take turns describing what stopped working, and the volunteers ask questions before reaching for a screwdriver. Some repairs succeed quickly; others become lessons in what to try next. Nobody is asked to pay, and nobody is hurried toward a perfect result. The goal is to help useful things last longer, share skills that might otherwise remain hidden, and make it easier for strangers to begin a conversation. At the end of the afternoon, the tables are cleared, the tools are counted, and a list of unfinished jobs is saved for next month.
        """.trimIndent().replace('\n', ' ')
    }
}
