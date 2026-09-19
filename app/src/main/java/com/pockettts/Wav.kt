package com.pockettts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** 24 kHz mono 16-bit WAV writer (adb-pullable). */
object Wav {
    fun write(file: File, audio: FloatArray, sampleRate: Int = PocketTtsSynthesizer.SAMPLE_RATE) {
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
