package com.pockettts

import android.util.Half
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import dev.pockettts.Accel
import dev.pockettts.Placement
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsConfig
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import dev.pockettts.Wav
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Random
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** A deterministic Tensor G5 precision probe; source and AOT graphs live in the model directory. */
@RunWith(AndroidJUnit4::class)
class FlowLmNpuPrecisionTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelDir get() = context.getExternalFilesDir(null) ?: context.filesDir

    private data class Inputs(val step: List<FloatArray>, val fused: List<FloatArray>, val noise: FloatArray)

    private fun inputs(): Inputs {
        val voice = ByteBuffer.wrap(File(modelDir, PocketTts.voiceFile("alba")).readBytes())
            .order(ByteOrder.LITTLE_ENDIAN)
        val pos = voice.int
        val perVoice = PocketTts.G * pos * PocketTts.HD
        val pk = FloatArray(PocketTts.G * PocketTts.PMAX * PocketTts.HD)
        val pv = FloatArray(pk.size)
        for (g in 0 until PocketTts.G) {
            val dst = g * PocketTts.PMAX * PocketTts.HD
            for (j in 0 until pos * PocketTts.HD) pk[dst + j] = Half.toFloat(voice.short)
        }
        for (g in 0 until PocketTts.G) {
            val dst = g * PocketTts.PMAX * PocketTts.HD
            for (j in 0 until pos * PocketTts.HD) pv[dst + j] = Half.toFloat(voice.short)
        }
        assertEquals(perVoice * 2, (voice.position() - Int.SIZE_BYTES) / Short.SIZE_BYTES)

        val emb = FloatArray(PocketTts.H)
        ByteBuffer.wrap(File(modelDir, PocketTts.BOS).readBytes())
            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(emb)
        val cosArr = FloatArray(PocketTts.HD)
        val sinArr = FloatArray(PocketTts.HD)
        for (j in 0 until PocketTts.HD / 2) {
            val angle = pos / Math.pow(PocketTts.THETA, j / 32.0)
            val c = cos(angle).toFloat()
            val s = sin(angle).toFloat()
            cosArr[j] = c; cosArr[j + PocketTts.HD / 2] = c
            sinArr[j] = s; sinArr[j + PocketTts.HD / 2] = s
        }
        val mask = FloatArray(PocketTts.NH * (PocketTts.PMAX + 1)) { PocketTts.MASK_NEG }
        for (h in 0 until PocketTts.NH) {
            val base = h * (PocketTts.PMAX + 1)
            for (p in 0 until pos) mask[base + p] = 0f
            mask[base + PocketTts.PMAX] = 0f
        }
        val rng = Random(42L)
        val noise = FloatArray(PocketTts.LDIM) {
            (rng.nextGaussian() * sqrt(PocketTts.TEMP.toDouble())).toFloat()
        }
        val step = listOf(emb, cosArr, sinArr, mask, pk, pv)
        return Inputs(step, step + listOf(noise), noise)
    }

    private fun runGraph(name: String, accelerator: Accelerator, args: List<FloatArray>, env: Environment?): List<FloatArray> {
        val path = File(modelDir, name)
        assertTrue("missing graph: $path", path.isFile)
        val t0 = System.nanoTime()
        val model = CompiledModel.create(path.absolutePath, CompiledModel.Options(accelerator), env)
        val t1 = System.nanoTime()
        try {
            val ins = model.createInputBuffers()
            val outs = model.createOutputBuffers()
            val t2 = System.nanoTime()
            try {
                args.forEachIndexed { i, data -> ins[i].writeFloat(data) }
                val t3 = System.nanoTime()
                var result = emptyList<FloatArray>()
                val runs = DoubleArray(3)
                var readMs = 0.0
                repeat(runs.size) { i ->
                    val start = System.nanoTime()
                    model.run(ins, outs)
                    runs[i] = (System.nanoTime() - start) / 1e6
                    val readStart = System.nanoTime()
                    result = outs.map { it.readFloat() }
                    readMs = (System.nanoTime() - readStart) / 1e6
                }
                Log.i(TAG, "graph=$name accelerator=$accelerator loadMs=${(t1 - t0) / 1e6} " +
                    "bufferAllocMs=${(t2 - t1) / 1e6} inputMs=${(t3 - t2) / 1e6} " +
                    "runMs=${runs.joinToString(",")} lastReadMs=$readMs")
                return result
            } finally {
                ins.forEach { it.close() }
                outs.forEach { it.close() }
            }
        } finally {
            model.close()
        }
    }

    private fun compare(label: String, cpu: FloatArray, npu: FloatArray) {
        assertEquals("$label tensor size", cpu.size, npu.size)
        var max = 0.0
        var sq = 0.0
        var dot = 0.0
        var aa = 0.0
        var bb = 0.0
        for (i in cpu.indices) {
            val a = cpu[i].toDouble()
            val b = npu[i].toDouble()
            assertTrue("$label nonfinite at $i", a.isFinite() && b.isFinite())
            val d = kotlin.math.abs(a - b)
            max = maxOf(max, d)
            sq += d * d
            dot += a * b
            aa += a * a
            bb += b * b
        }
        val corr = dot / (sqrt(aa * bb) + 1e-30)
        Log.i(TAG, "$label n=${cpu.size} maxAbs=$max rms=${sqrt(sq / cpu.size)} cosine=$corr")
    }

    @Test
    fun oneStepCpuVersusNoTruncationNpu() {
        val x = inputs()
        val env = Environment.create(context, mapOf(
            Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir,
        ))
        try {
            // Load NPU first: LiteRT dispatch settings are process-global.
            val npuStep = runGraph("pt_flowlm_step_fp16_no_truncation_g5.tflite", Accelerator.NPU, x.step, env)
            val cpuStep = runGraph("pt_flowlm_step_fp16.tflite", Accelerator.CPU, x.step, null)
            assertEquals(4, cpuStep.size)
            for (i in cpuStep.indices) compare("backbone output[$i]", cpuStep[i], npuStep[i])

            val cond = cpuStep.single { it.size == PocketTts.H }
            val headArgs = listOf(cond, x.noise)
            val npuHead = runGraph("pt_flow_head_fp16_no_truncation_g5.tflite", Accelerator.NPU, headArgs, env)
            val cpuHead = runGraph("pt_flow_head_fp16.tflite", Accelerator.CPU, headArgs, null)
            compare("head latent with identical CPU cond", cpuHead.single(), npuHead.single())

            val npuFused = runGraph("pt_flowlm_fused_fp16_no_truncation_g5.tflite", Accelerator.NPU, x.fused, env).single()
            val cpuFused = runGraph("pt_flowlm_fused_fp16.tflite", Accelerator.CPU, x.fused, null).single()
            val kv = PocketTts.G * PocketTts.HD
            compare("fused eos", cpuFused.copyOfRange(0, 1), npuFused.copyOfRange(0, 1))
            compare("fused latent", cpuFused.copyOfRange(1, 1 + PocketTts.LDIM),
                npuFused.copyOfRange(1, 1 + PocketTts.LDIM))
            compare("fused new-k", cpuFused.copyOfRange(1 + PocketTts.LDIM, 1 + PocketTts.LDIM + kv),
                npuFused.copyOfRange(1 + PocketTts.LDIM, 1 + PocketTts.LDIM + kv))
            compare("fused new-v", cpuFused.copyOfRange(1 + PocketTts.LDIM + kv, cpuFused.size),
                npuFused.copyOfRange(1 + PocketTts.LDIM + kv, npuFused.size))
        } finally {
            env.close()
        }
    }

    @Test
    fun shortSpeechCpuVersusNpuTruncationVariants() = compareSpeech(
        "Hello world. I am Pocket TTS running on a phone.", "short",
    )

    @Test
    fun longSpeechCpuVersusNpuTruncationVariants() = compareSpeech(LONG_PARAGRAPH, "long")

    private fun compareSpeech(phrase: String, label: String) {
        val models = PocketTtsModels.default(context)
        fun run(placement: Placement, graph: String) =
            PocketTtsEngine(context, PocketTtsConfig(
                models = models,
                placement = placement,
                lmGraph = graph,
                noiseSeed = 42L,
            )).use { it.stream(phrase, "alba") {} }
        val npuPlacement = Placement(Accel.NPU, Accel.NPU, Accel.GPU)
        val cpuPlacement = Placement(Accel.CPU, Accel.NPU, Accel.GPU)
        // NPU first: LiteRT dispatch settings are process-global.
        val noTrunc = run(npuPlacement, "pt_flowlm_fused_fp16_no_truncation.tflite")
        val sdkDefault = run(npuPlacement, "pt_flowlm_fused_fp16.tflite")
        val reference = run(cpuPlacement, "pt_flowlm_fused_fp16.tflite")
        val dir = context.getExternalFilesDir("flow-lm-npu") ?: context.filesDir
        dir.mkdirs()
        Wav.write(File(dir, "flowlm-$label-cpu-reference.wav"), reference.audio)
        for ((name, candidate) in listOf("no-truncation" to noTrunc, "sdk-default" to sdkDefault)) {
            Wav.write(File(dir, "flowlm-$label-$name.wav"), candidate.audio)
            val quality = AudioQuality.compare(reference.audio, candidate.audio)
            val audioSeconds = candidate.audio.size.toDouble() / PocketTts.SAMPLE_RATE
            Log.i(TAG, "speech=$label variant=$name cpuFrames=${reference.frames} " +
                "npuFrames=${candidate.frames} cpuMs=${reference.ms} npuMs=${candidate.ms} " +
                "npuRtf=${audioSeconds * 1000 / candidate.ms} corr=${quality.corr} " +
                "cpuLm=${reference.profile.lmRunMs} npuLm=${candidate.profile.lmRunMs}")
        }
    }

    private companion object {
        const val TAG = "FlowLmNpuPrecision"
        // Same 201-word paragraph used by PowerBenchmarkTest.
        val LONG_PARAGRAPH = """
            Each spring, a small group of neighbors meets at the public library to plan a weekend repair fair. They bring lamps with loose switches, radios that have gone quiet, bicycles with stubborn brakes, and kitchen tools that only need a little attention. Before the doors open, volunteers arrange the tables by task and place a handwritten sign beside every box of spare parts. A retired engineer shows the children how to trace a simple circuit, while a local baker sets out warm bread and explains how patient practice can turn a difficult recipe into an ordinary part of the day.

            By midmorning, the room is busy but calm. People take turns describing what stopped working, and the volunteers ask questions before reaching for a screwdriver. Some repairs succeed quickly; others become lessons in what to try next. Nobody is asked to pay, and nobody is hurried toward a perfect result. The goal is to help useful things last longer, share skills that might otherwise remain hidden, and make it easier for strangers to begin a conversation. At the end of the afternoon, the tables are cleared, the tools are counted, and a list of unfinished jobs is saved for next month.
        """.trimIndent().replace('\n', ' ')
    }
}
