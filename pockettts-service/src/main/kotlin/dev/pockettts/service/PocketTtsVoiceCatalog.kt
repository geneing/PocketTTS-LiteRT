package dev.pockettts.service

import android.content.Context
import dev.pockettts.PocketTtsModels
import dev.pockettts.Voice
import dev.pockettts.VoiceCatalog

/**
 * The voices the TTS service publishes: bundled presets whose cache is installed
 * plus user-generated `pt_voice_<name>.bin` files.
 *
 * Thin wrapper over [VoiceCatalog] so the service and the demo app cannot drift
 * apart on which voices exist; the discovery rules live there.
 */
internal object PocketTtsVoiceCatalog {

    fun installed(context: Context): List<Voice> =
        VoiceCatalog.installed(PocketTtsModels.default(context))

    fun named(installed: List<Voice>, name: String?): Voice? =
        VoiceCatalog.named(installed, name)
}
