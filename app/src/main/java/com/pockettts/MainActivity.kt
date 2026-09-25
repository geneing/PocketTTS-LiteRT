package com.pockettts

import android.app.Activity
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import dev.pockettts.PocketTts
import dev.pockettts.PocketTtsEngine
import dev.pockettts.PocketTtsModels
import dev.pockettts.VoiceCatalog
import dev.pockettts.Wav
import java.io.File
import java.util.concurrent.Executors

/**
 * Minimal Pocket TTS UI: pick a voice, type a sentence, tap Generate, listen.
 * All model work is the `:pockettts-core` library; this class only drives it and
 * plays the streamed chunks. Model load and generation run on a background
 * thread; audio plays via AudioTrack (float PCM) and the last output is saved to
 * filesDir/output.wav.
 */
class MainActivity : Activity() {

    private val bg = Executors.newSingleThreadExecutor()
    private var engine: PocketTtsEngine? = null

    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var voices: Spinner
    private lateinit var button: Button
    private lateinit var benchButton: Button
    private lateinit var waveform: WaveformView
    private var benchRuns = 3

    /**
     * The intent that drove the current launch. Held so the request can be applied
     * once the model finishes loading: [onCreate] starts that load on [bg] and the
     * initial [intent] is not necessarily what [onNewIntent] last saw.
     */
    private var pendingIntent: android.content.Intent? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            fitsSystemWindows = true
            setPadding(48, 48, 48, 48)
        }
        input = EditText(this).apply {
            hint = "Enter text to speak"
            setText("Hello! I am Pocket TTS, a tiny hundred million parameter model speaking to you from this phone.")
            minLines = 2
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                android.text.InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }
        voices = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                voiceNames(),
            )
        }
        button = Button(this).apply { text = "Generate"; isEnabled = false }
        benchButton = Button(this).apply { text = "Benchmark"; isEnabled = false }
        status = TextView(this).apply { text = "Loading model…"; textSize = 14f }
        waveform = WaveformView(this)
        val topMargins = intArrayOf(0, 24, 32, 8, 24)
        for ((index, view) in listOf(input, voices, button, benchButton, status).withIndex()) {
            val params = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            params.topMargin = topMargins[index]
            root.addView(view, params)
        }
        root.addView(waveform, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f).apply { topMargin = 24 })
        setContentView(root)

        pendingIntent = intent
        bg.execute {
            val e = try {
                PocketTtsEngine(this)
            } catch (e: Throwable) {
                android.util.Log.e("PocketTTS", "load failed", e)
                runOnUiThread { status.text = "Load failed: ${e.message}" }
                return@execute
            }
            engine = e
            android.util.Log.i("PocketTTS", "ready (${e.placements})")
            runOnUiThread {
                status.text = "Ready (${e.placements})."
                button.isEnabled = true
                benchButton.isEnabled = true
                runFromIntent(pendingIntent)
            }
        }

        button.setOnClickListener {
            val text = input.text.toString().ifBlank { return@setOnClickListener }
            val voice = voices.selectedItem as String
            button.isEnabled = false
            status.text = "Generating…"
            bg.execute {
                val e = engine ?: return@execute
                try {
                    // Streaming: play each chunk as the decoder produces it, so
                    // audio starts during generation. The library falls back to a
                    // single chunk when the smaller SEANet graph is not installed.
                    val track = streamTrack()
                    track.play()
                    val r = e.stream(text, voice) { chunk ->
                        track.write(chunk, 0, chunk.size, AudioTrack.WRITE_BLOCKING)
                    }
                    track.stop()
                    track.release()
                    saveWav(r.audio, voice)
                    val secs = r.audio.size.toFloat() / PocketTts.SAMPLE_RATE
                    val line = (
                        "Spoke %.1fs (%d frames) in %d ms wall — first audio %d ms, " +
                            "%d chunks (%s)"
                        ).format(
                        secs, r.frames, r.ms,
                        r.profile.firstChunkMs, r.profile.audioChunks, e.placements,
                    )
                    android.util.Log.i("PocketTTS", line)
                    runOnUiThread {
                        status.text = line
                        button.isEnabled = true
                        waveform.start(r.audio, PocketTts.SAMPLE_RATE)
                    }
                } catch (e: Throwable) {
                    android.util.Log.e("PocketTTS", "generation failed", e)
                    runOnUiThread { status.text = "Error: ${e.message}"; button.isEnabled = true }
                }
            }
        }
        benchButton.setOnClickListener {
            val text = input.text.toString().ifBlank { return@setOnClickListener }
            val voice = voices.selectedItem as String
            button.isEnabled = false
            benchButton.isEnabled = false
            status.text = "Benchmarking… (logcat tag PocketTTSBench)"
            val runs = benchRuns
            bg.execute {
                // Free the UI model first: each Benchmarker placement loads its own.
                engine?.close(); engine = null
                try {
                    Benchmarker(this).run(text, voice, runs)
                } catch (e: Throwable) {
                    android.util.Log.e("PocketTTS", "benchmark failed", e)
                }
                val e = try { PocketTtsEngine(this) } catch (e: Throwable) { null }
                engine = e
                runOnUiThread {
                    status.text = if (e != null) {
                        "Benchmark done — see benchmark.txt / logcat. Ready (${e.placements})."
                    } else {
                        "Benchmark done, but model reload failed."
                    }
                    button.isEnabled = e != null
                    benchButton.isEnabled = true
                }
            }
        }
    }

    /**
     * Headless driving:
     *   adb shell am start -n com.pockettts/.MainActivity --es text "hi" --es voice alba
     *   adb shell am start -n com.pockettts/.MainActivity --ez bench true --ei runs 3
     * (singleTop, so a second am start re-runs without reloading the model.)
     *
     * The voice list is rebuilt per call from what is installed, so a cache pushed
     * while the app is running becomes selectable without a restart.
     */
    private fun voiceNames(): List<String> =
        VoiceCatalog.installed(PocketTtsModels.default(this)).map { it.name }

    private fun runFromIntent(i: android.content.Intent?) {
        if (i == null) return
        i.getStringExtra("text")?.let { t ->
            input.setText(t)
            i.getStringExtra("voice")?.let { v ->
                val names = voiceNames()
                val idx = names.indexOfFirst { it.equals(v, ignoreCase = true) }
                if (idx >= 0) {
                    (voices.adapter as ArrayAdapter<String>).apply {
                        clear()
                        addAll(names)
                        notifyDataSetChanged()
                    }
                    voices.setSelection(idx)
                } else {
                    android.util.Log.w("PocketTTS", "voice '$v' not installed; have $names")
                }
            }
        }
        if (i.getBooleanExtra("bench", false)) {
            benchRuns = i.getIntExtra("runs", 3)
            if (benchButton.isEnabled) benchButton.performClick()
            return
        }
        // Not gated on button.isEnabled: on a cold start this runs from the same
        // UI callback that enables it, and the engine is what actually gates it.
        if (i.hasExtra("text")) button.performClick()
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        pendingIntent = intent
        runFromIntent(intent)
    }

    /** Save the last output as a 24 kHz mono 16-bit WAV in filesDir (adb-pullable). */
    private fun saveWav(audio: FloatArray, voice: String) {
        Wav.write(File(filesDir, "output.wav"), audio)
        Wav.write(File(filesDir, "output_$voice.wav"), audio)
    }

    /**
     * An AudioTrack in streaming mode. The buffer covers the gap between one
     * SEANet window and the next; the blocking writes throttle the decoder to
     * playback rate rather than letting generated audio pile up.
     */
    private fun streamTrack(): AudioTrack {
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .build()
        val fmt = AudioFormat.Builder()
            .setSampleRate(PocketTts.SAMPLE_RATE)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        val min = AudioTrack.getMinBufferSize(
            PocketTts.SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        val bytes = maxOf(min, PocketTts.SAMPLE_RATE * 4 * 2)
        return AudioTrack(
            attrs, fmt, bytes,
            AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdownNow()
        engine?.close()
    }
}
