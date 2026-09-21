package dev.pockettts

import android.content.Context
import java.io.File

/**
 * Everything that shapes an engine: where the files come from, where each graph
 * runs, and the decode knobs. Immutable; one config per [PocketTtsEngine].
 */
class PocketTtsConfig(
    /** Where model files are resolved from. */
    val models: PocketTtsModels,
    /** Per-graph accelerator choice. */
    val placement: Placement,
    /** Override the flow-LM graph filename (e.g. a quantized variant). null = auto. */
    val lmGraph: String? = null,
    /** Frames per LM invocation; >1 needs the matching `pt_flowlm_ms{N}` graph. */
    val lmSteps: Int = 1,
    /** SEANet window (feature positions) for streaming. */
    val streamW: Int = PocketTts.STREAM_W,
    /** When set, every synthesis reseeds the noise RNG so repeats are identical. */
    val noiseSeed: Long? = null,
    /** When set, GPU graphs serialize their compiled program cache here. */
    val gpuCache: File? = null,
) {
    companion object {
        /**
         * The device policy: adb-pushed models first, GitHub release fallback,
         * and [Placement.default] for the accelerator split.
         */
        fun default(
            context: Context,
            models: PocketTtsModels = PocketTtsModels.default(context),
        ): PocketTtsConfig {
            val dir = context.getExternalFilesDir(null) ?: context.filesDir
            return PocketTtsConfig(models, Placement.default(context, dir))
        }
    }
}
