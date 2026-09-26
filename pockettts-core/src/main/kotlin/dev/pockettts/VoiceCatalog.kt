package dev.pockettts

/**
 * Discovers the voices an engine can actually speak: the bundled presets, plus
 * any cache fetched or cloned into the `voices/` subdirectory (see
 * `scripts/download_voices.py` and `scripts/create_voice.py`).
 *
 * This is the single place that answers "what voices exist". The bundled list
 * ([Voice.all]) is only a starting point: a voice cache is a first-class voice
 * with no Android build change, and a bundled name whose file is absent is not
 * offered.
 *
 * Lives in `:pockettts-core` rather than the service because the demo app and the
 * engine defaults need the same answer; `PocketTtsVoiceCatalog` in
 * `:pockettts-service` delegates here.
 */
object VoiceCatalog {

    /** A voice name a cache file may use, matching the Python writer's rule. */
    internal val SAFE_NAME = Regex("[a-z][a-z0-9_-]{0,63}")

    /** Where a fetched/cloned cache lives, relative to the model root. */
    private const val DIR = PocketTts.VOICES_DIR + "/"

    /**
     * Names worth probing for in `voices/`. Under scoped storage an app cannot
     * reliably enumerate a subdirectory adb created, so discovery has to ask
     * about known names. This is the set the project's scripts install; a cache
     * with any other name is still found through [namesIn] where enumeration
     * works (a bundled asset dir, a release zip, or an app-owned root).
     */
    private val KNOWN = listOf(
        "anna", "azelma", "bill_boerst", "caro_davy", "cosette", "daan",
        "eponine", "estelle", "fantine", "george", "giovanni", "jane", "jean",
        "juergen", "lola", "michael", "paul", "peter_yearsley", "rafael",
        "stuart_bell", "vera",
    )

    /**
     * The voices [models] can speak: bundled presets first in their published
     * order, then extra caches, sorted by name. Falls back to [Voice.all] when
     * nothing is installed, so a config is never empty.
     */
    fun installed(models: PocketTtsModels): List<Voice> {
        val bundled = Voice.all()
        val shipped = bundled.filter { models.store.exists(PocketTts.voiceFile(it.name)) }
        val custom = customVoices(models, bundled.map { it.name })
        return (shipped + custom).ifEmpty { Voice.all() }
    }

    /**
     * Create the `voices/` directory under the app's writable model root.
     *
     * Necessary because of how the directory gets populated: adb creates it owned
     * by `shell` with `drwxrws---`, which the app process cannot even traverse, so
     * files pushed in are invisible to it. Creating it from inside the app makes it
     * app-owned, after which `adb push` into the existing directory works. Call
     * this before an install script pushes voices, or once on startup.
     */
    fun ensureDir(models: PocketTtsModels): java.io.File? =
        models.store.locate(PocketTts.EMBED)?.parentFile
            ?.let { java.io.File(it, PocketTts.VOICES_DIR) }
            ?.also { it.mkdirs() }

    /**
     * Extra caches that are not bundled names, sorted.
     *
     * Both layouts are read: `voices/` is where the scripts write, and the model
     * root is where caches pushed before that convention sit. A name present in
     * both is one voice.
     *
     * Discovery probes rather than enumerates: under scoped storage an app cannot
     * reliably `listFiles()` a subdirectory that adb created, but `locate` on a
     * known name works from either layout. So the candidate set is the bundled
     * names plus everything ever installed here, and existence decides.
     */
    private fun customVoices(models: PocketTtsModels, known: List<String>): List<Voice> {
        val candidates = LinkedHashSet<String>()
        candidates += namesIn(models, DIR)
        candidates += namesIn(models, "")
        candidates += knownNames()
        return candidates
            .filter { name -> known.none { it.equals(name, ignoreCase = true) } }
            .filter { fileFor(models, it) != null }
            .sorted()
            .map { Voice(it) }
    }

    /** Names installed by earlier runs, so a probe-only store can still find them. */
    private fun knownNames(): List<String> = KNOWN.toList()

    private fun namesIn(models: PocketTtsModels, dir: String): List<String> =
        models.store.list("$dir${PocketTts.VOICE_PREFIX}", PocketTts.VOICE_SUFFIX)
            .mapNotNull { it.voiceNameFromFile() }

    /**
     * `pt_voice_<name>.bin` -> `<name>`, or null when the name is unusable.
     * Callers pass a filename from one directory, so no path is expected here.
     */
    internal fun String.voiceNameFromFile(): String? {
        if (!startsWith(PocketTts.VOICE_PREFIX) || !endsWith(PocketTts.VOICE_SUFFIX)) return null
        val stem = removePrefix(PocketTts.VOICE_PREFIX).removeSuffix(PocketTts.VOICE_SUFFIX)
        return stem.takeIf { SAFE_NAME.matches(it) }
    }

    /**
     * The file backing [name]: `voices/` first, then the model root, mirroring the
     * install layout. Null when the voice is not installed.
     */
    fun fileFor(models: PocketTtsModels, name: String): java.io.File? =
        models.store.locate(DIR + PocketTts.voiceFile(name))
            ?: models.store.locate(PocketTts.voiceFile(name))

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
