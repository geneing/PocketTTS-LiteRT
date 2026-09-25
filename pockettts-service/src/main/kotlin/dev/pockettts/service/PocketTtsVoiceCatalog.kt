package dev.pockettts.service

import android.content.Context
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsModels
import dev.pockettts.Voice
import java.io.File

/** Discovers shipped voices plus user-generated ``pt_voice_<name>.bin`` files. */
internal object PocketTtsVoiceCatalog {

    private val safeName = Regex("[a-z][a-z0-9_-]{0,63}")

    fun installed(context: Context): List<Voice> {
        val models = PocketTtsModels.default(context)
        val shipped = Voice.all().filter { models.store.exists(PocketTts.voiceFile(it.name)) }
        val modelDir = context.getExternalFilesDir(null) ?: context.filesDir
        val custom = modelDir.listFiles()
            .orEmpty()
            .mapNotNull { it.customVoiceName() }
            .filter { name -> Voice.all().none { it.name == name } }
            .distinct()
            .sorted()
            .map { Voice(it) }

        return (shipped + custom).ifEmpty { Voice.all() }
    }

    fun named(installed: List<Voice>, name: String?): Voice? {
        val bare = name?.trim()?.lowercase()?.substringBefore('#')
            ?.removePrefix("pockettts-") ?: return null
        return installed.firstOrNull { it.name.equals(bare, ignoreCase = true) }
    }

    private fun File.customVoiceName(): String? {
        if (!isFile || length() < 4 || !name.startsWith("pt_voice_") || !name.endsWith(".bin")) {
            return null
        }
        val stem = name.removePrefix("pt_voice_").removeSuffix(".bin")
        if (!safeName.matches(stem)) return null

        // Reject corrupt or unrelated files before advertising them to Android.
        val length = inputStream().buffered().use { input ->
            val header = ByteArray(4)
            if (input.read(header) != header.size) return null
            java.nio.ByteBuffer.wrap(header).order(java.nio.ByteOrder.LITTLE_ENDIAN).int
        }
        // The file has both K and V, each stored as 16-bit values.
        val expected = 4L + 4L * PocketTts.G * length * PocketTts.HD
        return stem.takeIf { length in 1..PocketTts.PMAX && this.length() == expected }
    }
}
