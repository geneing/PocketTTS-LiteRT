package dev.pockettts

/**
 * Discovers the voices an engine can actually speak: the bundled presets whose
 * `pt_voice_<name>.bin` is resolvable through the model store, plus any
 * user-generated cache sitting next to it (see `scripts/create_voice.py`).
 *
 * This is the single place that answers "what voices exist". The bundled list
 * ([Voice.all]) is only a starting point: a cache produced by the voice-cloning
 * script is a first-class voice with no Android build change, and a bundled name
 * whose file is absent is not offered.
 *
 * Lives in `:pockettts-core` rather than the service because the demo app and the
 * engine defaults need the same answer; [PocketTtsVoiceCatalog] in
 * `:pockettts-service` delegates here.
 */
object VoiceCatalog {

    /** A voice name a cache file may use, matching the Python writer's rule. */
    internal val SAFE_NAME = Regex("[a-z][a-z0-9_-]{0,63}")

    /**
     * The voices [models] can speak, bundled first in their published order and
     * custom caches after, sorted by name. Falls back to [Voice.all] when nothing
     * is installed, so a config is never empty.
     */
    fun installed(models: PocketTtsModels): List<Voice> {
        val shipped = Voice.all().filter { models.store.exists(PocketTts.voiceFile(it.name)) }
        val custom = customVoices(models, shipped.map { it.name })
        return (shipped + custom).ifEmpty { Voice.all() }
    }

    /**
     * Custom caches the store can see that are not already [known] bundled names.
     *
     * Resolved through the store rather than by listing one directory, so a cache
     * served from a bundled asset directory or a release zip is found the same way
     * a pushed one is.
     */
    private fun customVoices(models: PocketTtsModels, known: List<String>): List<Voice> =
        models.store.list(PocketTts.VOICE_PREFIX, PocketTts.VOICE_SUFFIX)
            .mapNotNull { it.voiceNameFromFile() }
            .filter { name -> known.none { it.equals(name, ignoreCase = true) } }
            .distinct()
            .sorted()
            .map { Voice(it) }

    /** `pt_voice_<name>.bin` -> `<name>`, or null when the file is not a valid cache. */
    internal fun String.voiceNameFromFile(): String? {
        if (!startsWith(PocketTts.VOICE_PREFIX) || !endsWith(PocketTts.VOICE_SUFFIX)) return null
        val stem = removePrefix(PocketTts.VOICE_PREFIX).removeSuffix(PocketTts.VOICE_SUFFIX)
        return stem.takeIf { SAFE_NAME.matches(it) }
    }

    /**
     * Resolve [name] against [installed]: accepts the bare name, a
     * `pockettts-`-prefixed cross-engine id, and the `#`-suffixed forms some
     * clients send. Null when nothing matches.
     */
    fun named(installed: List<Voice>, name: String?): Voice? {
        val bare = name?.trim()?.lowercase()?.substringBefore('#')
            ?.removePrefix(PocketTts.VOICE_ID_PREFIX) ?: return null
        return installed.firstOrNull { it.name.equals(bare, ignoreCase = true) }
    }
}
