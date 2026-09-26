package dev.pockettts

import android.content.Context
import android.content.res.AssetManager
import org.json.JSONObject
import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/**
 * A place model files can come from. Implementations return the *file* for a
 * name when it is available, or null when this source does not (yet) have it.
 * A source may materialize a file as a side effect (asset copy, download).
 */
interface ModelSource {
    fun locate(name: String): File?

    /**
     * Names this source can serve that start with [prefix] and end with [suffix].
     *
     * Optional: sources that cannot enumerate their contents (a bundled asset
     * list, a release index) return an empty list rather than a wrong answer.
     * Used by [VoiceCatalog] to find user-generated voice caches. A [prefix] may
     * include a subdirectory (`voices/pt_voice_`), which a file-backed source
     * resolves relative to itself.
     */
    fun list(prefix: String, suffix: String): List<String> = emptyList()
}

/** Files already on disk - the external files dir (adb push) or an app dir. */
class DirectorySource(private val dir: File) : ModelSource {
    override fun locate(name: String): File? = File(dir, name).takeIf { it.isFile }

    override fun list(prefix: String, suffix: String): List<String> {
        // The prefix may name a subdirectory; enumerate that instead of the root.
        val cut = prefix.lastIndexOf('/')
        val root = if (cut < 0) dir else File(dir, prefix.substring(0, cut))
        val stem = if (cut < 0) prefix else prefix.substring(cut + 1)
        return root.listFiles().orEmpty().map { it.name }
            .filter { it.startsWith(stem) && it.endsWith(suffix) }
    }
}

/**
 * Files bundled in the APK's assets (optionally under [assetPrefix]). Assets are
 * not mmap-able, so the first lookup copies the file into [cacheDir].
 */
class AssetModelSource(
    private val assets: AssetManager,
    private val cacheDir: File,
    private val assetPrefix: String = "",
) : ModelSource {
    override fun locate(name: String): File? {
        val dst = File(cacheDir, name)
        if (dst.isFile) return dst
        val stream: InputStream = try {
            assets.open(assetPrefix + name)
        } catch (e: Exception) {
            return null
        }
        cacheDir.mkdirs()
        val tmp = File(cacheDir, "$name.part")
        stream.use { i -> tmp.outputStream().use { o -> i.copyTo(o) } }
        if (!tmp.renameTo(dst)) {
            tmp.copyTo(dst, overwrite = true)
            tmp.delete()
        }
        return dst
    }
}

/** First source that has the file wins. Order is priority. */
class CompositeModelSource(private val sources: List<ModelSource>) : ModelSource {
    override fun locate(name: String): File? {
        for (s in sources) s.locate(name)?.let { return it }
        return null
    }
}

/** A versioned bundle of model files published as one release asset zip. */
data class ModelVariant(
    val name: String,
    val zip: String,
    val sha256: String,
    val bytes: Long,
    val files: List<String>,
)

/**
 * `models.json`, published alongside the zips. Maps each file to the variant
 * that carries it so the downloader only fetches what the placement needs.
 */
class ModelManifest(
    val modelVersion: String,
    val modelApi: Int,
    val variants: List<ModelVariant>,
) {
    fun variantFor(file: String): ModelVariant? =
        variants.firstOrNull { file in it.files }

    companion object {
        fun parse(json: String): ModelManifest {
            val o = JSONObject(json)
            val arr = o.optJSONArray("variants") ?: org.json.JSONArray()
            val variants = ArrayList<ModelVariant>(arr.length())
            for (i in 0 until arr.length()) {
                val v = arr.getJSONObject(i)
                val filesArr = v.optJSONArray("files") ?: org.json.JSONArray()
                val files = ArrayList<String>(filesArr.length())
                for (j in 0 until filesArr.length()) files.add(filesArr.getString(j))
                variants.add(
                    ModelVariant(
                        name = v.getString("name"),
                        zip = v.getString("zip"),
                        sha256 = v.getString("sha256"),
                        bytes = v.optLong("bytes", 0L),
                        files = files,
                    ),
                )
            }
            return ModelManifest(o.getString("modelVersion"), o.optInt("modelApi", 1), variants)
        }

        fun fetch(url: String, downloader: Downloader): ModelManifest =
            parse(String(downloader.read(url), Charsets.UTF_8))
    }
}

/** Pluggable transport so the library does not force an HTTP stack on consumers. */
interface Downloader {
    /** Write [url] to [dst] (resuming when the server supports it). */
    fun download(url: String, dst: File, onProgress: (Long, Long) -> Unit)

    fun read(url: String): ByteArray
}

/** Dependency-free default backed by [HttpURLConnection]. */
class HttpDownloader(
    private val userAgent: String = "PocketTTS-LiteRT",
) : Downloader {

    override fun download(url: String, dst: File, onProgress: (Long, Long) -> Unit) {
        val tmp = File(dst.parentFile, "${dst.name}.part")
        val have = if (tmp.isFile) tmp.length() else 0L
        val conn = open(url)
        if (have > 0) conn.setRequestProperty("Range", "bytes=$have-")
        val code = conn.responseCode
        val resuming = have > 0 && code == HttpURLConnection.HTTP_PARTIAL
        val total = conn.contentLengthLong.let { if (it <= 0) -1L else it + if (resuming) have else 0L }
        tmp.parentFile?.mkdirs()
        conn.inputStream.use { input ->
            java.io.FileOutputStream(tmp, resuming).use { output ->
                val buf = ByteArray(1 shl 20)
                var written = if (resuming) have else 0L
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    output.write(buf, 0, n)
                    written += n
                    onProgress(written, total)
                }
            }
        }
        if (!tmp.renameTo(dst)) {
            tmp.copyTo(dst, overwrite = true)
            tmp.delete()
        }
    }

    override fun read(url: String): ByteArray =
        open(url).inputStream.use { it.readBytes() }

    private fun open(url: String): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 60_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", userAgent)
        }
}

/**
 * Models published as assets on a GitHub release. [locate] only sees files that
 * have already been installed; [ensure] downloads and verifies the variants
 * needed for the requested names.
 *
 * @param releaseBase e.g.
 *   `https://github.com/<owner>/<repo>/releases/download/models-2026.09`
 */
class ReleaseModelSource(
    private val releaseBase: String,
    private val cacheDir: File,
    private val downloader: Downloader = HttpDownloader(),
    private val preloadedManifest: ModelManifest? = null,
) : ModelSource {

    @Volatile
    private var manifest: ModelManifest? = preloadedManifest

    override fun locate(name: String): File? = File(cacheDir, name).takeIf { it.isFile }

    /** Download whichever variants contain the missing [names]. */
    fun ensure(names: Collection<String>, onProgress: (Long, Long) -> Unit = { _, _ -> }) {
        val missing = names.filter { locate(it) == null }
        if (missing.isEmpty()) return
        val man = manifest ?: ModelManifest.fetch("$releaseBase/models.json", downloader)
            .also { manifest = it }
        val variants = missing.mapNotNull { man.variantFor(it) }.distinct()
        check(variants.isNotEmpty()) { "no release variant provides ${missing.joinToString()}" }
        val total = variants.sumOf { if (it.bytes > 0) it.bytes else 0L }
        var done = 0L
        for (v in variants) {
            val zip = File(cacheDir, v.zip)
            if (!zip.isFile || sha256(zip) != v.sha256) {
                downloader.download("$releaseBase/${v.zip}", zip) { d, _ ->
                    onProgress(done + d, if (total > 0) total else d)
                }
                check(sha256(zip) == v.sha256) { "sha256 mismatch for ${v.zip}" }
            }
            extract(zip, cacheDir, v.files)
            done += if (v.bytes > 0) v.bytes else zip.length()
            onProgress(done, if (total > 0) total else done)
        }
        val still = names.filter { locate(it) == null }
        check(still.isEmpty()) { "still missing after download: ${still.joinToString()}" }
    }

    /** Copy the listed entries (or all when empty) out of a stored zip. */
    private fun extract(zip: File, dir: File, wanted: List<String>) {
        dir.mkdirs()
        ZipInputStream(zip.inputStream().buffered()).use { zin ->
            while (true) {
                val e = zin.nextEntry ?: break
                if (e.isDirectory) continue
                if (wanted.isNotEmpty() && e.name !in wanted) continue
                val dst = File(dir, e.name)
                dst.parentFile?.mkdirs()
                val tmp = File(dir, "${e.name}.part")
                tmp.outputStream().use { o -> zin.copyTo(o) }
                if (!tmp.renameTo(dst)) {
                    tmp.copyTo(dst, overwrite = true)
                    tmp.delete()
                }
            }
        }
    }
}

/** Resolves each model file through an ordered list of sources. */
class ModelStore(private val sources: List<ModelSource>) {

    fun locate(name: String): File? {
        for (s in sources) s.locate(name)?.let { return it }
        return null
    }

    fun exists(name: String): Boolean = locate(name) != null

    /** Names any source can serve matching [prefix]/[suffix], deduplicated. */
    fun list(prefix: String, suffix: String): List<String> =
        sources.flatMap { it.list(prefix, suffix) }.distinct()

    fun file(name: String): File =
        locate(name) ?: throw FileNotFoundException(
            "Missing model file '$name'. Push models to the device " +
                "(scripts/install_to_device.sh) or call PocketTtsModels.ensure().",
        )

    companion object {
        fun of(vararg sources: ModelSource) = ModelStore(sources.toList())
    }
}

/**
 * The model set the engine reads: an adb-pushed / app-owned directory first,
 * then a GitHub release for cold installs. `DirectorySource` winning is what
 * makes on-device model iteration instant.
 */
class PocketTtsModels private constructor(
    val store: ModelStore,
    val release: ReleaseModelSource?,
) {
    /** Download anything in [names] the release has and the device does not. */
    fun ensure(names: Collection<String>, onProgress: (Long, Long) -> Unit = { _, _ -> }) {
        release?.ensure(names, onProgress)
            ?: throw IllegalStateException("this PocketTtsModels has no release source")
    }

    companion object {
        const val DEFAULT_VERSION = "2026.09"
        const val GITHUB_REPO = "geneing/PocketTTS-LiteRT"

        fun releaseBase(version: String, repo: String = GITHUB_REPO) =
            "https://github.com/$repo/releases/download/models-$version"

        /**
         * Directory source (used during development) + GitHub release fallback.
         * The directory is the app's external files dir, so `adb push` and
         * `install_to_device.sh` keep working unchanged.
         */
        fun default(
            context: Context,
            version: String = DEFAULT_VERSION,
            repo: String = GITHUB_REPO,
        ): PocketTtsModels {
            val dir = context.getExternalFilesDir(null) ?: context.filesDir
            val release = ReleaseModelSource(
                releaseBase = releaseBase(version, repo),
                cacheDir = File(dir, "models/$version"),
                downloader = HttpDownloader(),
            )
            return PocketTtsModels(
                ModelStore(listOf(DirectorySource(dir), release)),
                release,
            )
        }

        fun of(vararg sources: ModelSource) =
            PocketTtsModels(ModelStore(sources.toList()), null)

        /**
         * A release source as the model set, optionally preceded by local
         * sources. Unlike [of] this keeps the download capability, which is how
         * an app does a cold install (or an on-device directory override).
         */
        fun ofRelease(release: ReleaseModelSource, vararg localSources: ModelSource) =
            PocketTtsModels(
                ModelStore(listOf(*localSources, release)),
                release,
            )

        fun ofDirectory(dir: File) = of(DirectorySource(dir))
    }
}

internal fun sha256(file: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered().use { input ->
        val buf = ByteArray(1 shl 20)
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            md.update(buf, 0, n)
        }
    }
    return md.digest().joinToString("") { "%02x".format(it) }
}
