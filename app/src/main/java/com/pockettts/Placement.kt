package com.pockettts

import android.content.Context
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.GLES20
import java.io.File

/** Where one graph runs: GPU (fp16 weights/compute), GPU at fp32 compute, or CPU (XNNPACK). */
enum class Accel(val tag: String) {
    GPU("GPU"),
    CPU("CPU"),
    GPU32("GPU32"),
}

/**
 * Per-graph accelerator choice. Every graph can be placed independently; the
 * repo's measurements show the best split is device-specific (Mali/Adreno win
 * with the flow-LM on GPU, PowerVR loses badly there).
 */
data class Placement(val lm: Accel, val dectx: Accel, val deconly: Accel) {
    val label: String get() = "lm:${lm.tag} dectx:${dectx.tag} dec:${deconly.tag}"

    override fun toString(): String = label

    companion object {
        /** The reference: every graph on CPU. Used as the audio-quality gold. */
        val GOLD = Placement(Accel.CPU, Accel.CPU, Accel.CPU)

        /**
         * `GL_RENDERER` of the GPU the LiteRT delegate will use, e.g.
         * "PowerVR DXT-48-1536", "Mali-G715", "Adreno (TM) 740". Empty when no
         * EGL context can be created (then the GPU default is kept).
         */
        fun renderer(): String = try {
            val d = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val ver = IntArray(2)
            EGL14.eglInitialize(d, ver, 0, ver, 1)
            val cfg = arrayOfNulls<EGLConfig>(1)
            val n = IntArray(1)
            val attribs = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE,
            )
            EGL14.eglChooseConfig(d, attribs, 0, cfg, 0, cfg.size, n, 0)
            val ctx = EGL14.eglCreateContext(
                d, cfg[0], EGL14.EGL_NO_CONTEXT,
                intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0,
            )
            val surf = EGL14.eglCreatePbufferSurface(
                d, cfg[0],
                intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0,
            )
            EGL14.eglMakeCurrent(d, surf, surf, ctx)
            val r = GLES20.glGetString(GLES20.GL_RENDERER) ?: ""
            EGL14.eglMakeCurrent(
                d, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT,
            )
            EGL14.eglDestroySurface(d, surf)
            EGL14.eglDestroyContext(d, ctx)
            EGL14.eglTerminate(d)
            r
        } catch (t: Throwable) {
            ""
        }

        /** Parse the force_* override files and apply the device default. */
        fun default(context: Context): Placement {
            val dir = context.getExternalFilesDir(null)
            fun keys(file: String): Set<String> =
                File(dir, file).takeIf { it.exists() }
                    ?.readText()?.split(',', '\n')?.map { it.trim() }?.filter { it.isNotEmpty() }
                    ?.toSet() ?: emptySet()

            val userCpu = keys("force_cpu.txt")
            val fp32 = keys("force_fp32.txt")
            val forceGpu = keys("force_gpu.txt")
            val powerVr = renderer().contains("PowerVR", ignoreCase = true)

            // PowerVR (Tensor G5 / Pixel 10): the OpenCL delegate's per-AR-step
            // overhead (~500 dispatches plus a 25 MB packed-KV upload and a
            // 12.6 MB readback) makes the flow-LM ~2.7x SLOWER than XNNPACK
            // here — 0.81x vs 1.33x real-time, measured. Mali and Adreno still
            // win with the LM on GPU, so only PowerVR is auto-pinned to CPU.
            val lm = when {
                "lm" in userCpu -> Accel.CPU
                powerVr && "lm" !in forceGpu -> Accel.CPU
                "lm" in fp32 -> Accel.GPU32
                else -> Accel.GPU
            }
            // The Mimi decoder transformer defaults to CPU everywhere: its GPU
            // output is audibly degraded (Mali and confirmed on PowerVR), and
            // fp32 does not recover it. force_gpu.txt "dectx" re-enables GPU.
            val dectx = when {
                "dectx" !in forceGpu -> Accel.CPU
                "dectx" in fp32 -> Accel.GPU32
                else -> Accel.GPU
            }
            val deconly = when {
                "dec" in userCpu -> Accel.CPU
                "dec" in fp32 -> Accel.GPU32
                else -> Accel.GPU
            }
            return Placement(lm, dectx, deconly)
        }
    }
}
