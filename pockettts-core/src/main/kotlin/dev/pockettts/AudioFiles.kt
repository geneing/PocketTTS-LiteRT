package dev.pockettts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** PCM helpers for the float output of [PocketTtsEngine]. */
object Pcm {

    /** 24 kHz mono float in [-1, 1] -> little-endian 16-bit PCM. */
    fun floatToPcm16(audio: FloatArray): ByteArray {
        val out = ByteArray(audio.size * 2)
        var o = 0
        for (v in audio) {
            val s = (v.coerceIn(-1f, 1f) * 32767f).toInt().toShort()
            out[o++] = (s.toInt() and 0xFF).toByte()
            out[o++] = ((s.toInt() shr 8) and 0xFF).toByte()
        }
        return out
    }

    /** Same, into a caller-owned buffer (avoids a per-chunk allocation). */
    fun floatToPcm16(audio: FloatArray, dst: ByteArray) {
        var o = 0
        for (v in audio) {
            val s = (v.coerceIn(-1f, 1f) * 32767f).toInt().toShort()
            dst[o++] = (s.toInt() and 0xFF).toByte()
            dst[o++] = ((s.toInt() shr 8) and 0xFF).toByte()
        }
    }
}

/** 24 kHz mono 16-bit WAV writer. */
object Wav {
    fun write(file: File, audio: FloatArray, sampleRate: Int = PocketTts.SAMPLE_RATE) {
        val data = audio.size * 2
        val bb = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(36 + data); bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()); bb.putInt(16); bb.putShort(1); bb.putShort(1)
        bb.putInt(sampleRate); bb.putInt(sampleRate * 2); bb.putShort(2); bb.putShort(16)
        bb.put("data".toByteArray()); bb.putInt(data)
        for (v in audio) bb.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        file.writeBytes(bb.array())
    }
}
