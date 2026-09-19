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
import java.io.File
import java.util.concurrent.Executors

/**
 * Minimal Pocket TTS UI: pick a voice, type a sentence, tap Generate, listen.
 * Model load and generation run on a background thread; audio plays via
 * AudioTrack (float PCM) and the last output is saved to filesDir/output.wav.
 */
class MainActivity : Activity() {

    private val bg = Executors.newSingleThreadExecutor()
    private var synth: PocketTtsSynthesizer? = null

    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var voices: Spinner
    private lateinit var button: Button
    private lateinit var benchButton: Button
    private lateinit var waveform: WaveformView
    private var benchRuns = 3

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
                PocketTtsSynthesizer.VOICES,
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

        bg.execute {
            val s = try {
                PocketTtsSynthesizer(this)
            } catch (e: Throwable) {
                android.util.Log.e("PocketTTS", "load failed", e)
                runOnUiThread { status.text = "Load failed: ${e.message}" }
                return@execute
            }
            synth = s
            android.util.Log.i("PocketTTS", "ready (${s.placements})")
            runOnUiThread {
                status.text = "Ready (${s.placements})."
                button.isEnabled = true
                benchButton.isEnabled = true
                runFromIntent(intent)
            }
        }

        button.setOnClickListener {
            val text = input.text.toString().ifBlank { return@setOnClickListener }
            val voice = voices.selectedItem as String
            button.isEnabled = false
            status.text = "Generating…"
            bg.execute {
                val s = synth ?: return@execute
                try {
                    val r = s.synthesize(text, voice)
                    saveWav(r.audio, voice)
                    val secs = r.audio.size.toFloat() / PocketTtsSynthesizer.SAMPLE_RATE
                    val rtf = secs * 1000f / r.ms
                    val line = "Spoke %.1fs (%d frames) in %d ms — %.2fx real-time (%s)"
                        .format(secs, r.frames, r.ms, rtf, s.placements)
                    android.util.Log.i("PocketTTS", line)
                    runOnUiThread {
                        status.text = line
                        button.isEnabled = true
                        waveform.start(r.audio, PocketTtsSynthesizer.SAMPLE_RATE)
                    }
                    play(r.audio)
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
                synth?.close(); synth = null
                try {
                    Benchmarker(this).run(text, voice, runs)
                } catch (e: Throwable) {
                    android.util.Log.e("PocketTTS", "benchmark failed", e)
                }
                val s = try { PocketTtsSynthesizer(this) } catch (e: Throwable) { null }
                synth = s
                runOnUiThread {
                    status.text = if (s != null) {
                        "Benchmark done — see benchmark.txt / logcat. Ready (${s.placements})."
                    } else {
                        "Benchmark done, but model reload failed."
                    }
                    button.isEnabled = s != null
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
     */
    private fun runFromIntent(i: android.content.Intent?) {
        if (i == null) return
        i.getStringExtra("text")?.let { t ->
            input.setText(t)
            i.getStringExtra("voice")?.let { v ->
                val idx = PocketTtsSynthesizer.VOICES.indexOf(v)
                if (idx >= 0) voices.setSelection(idx)
            }
        }
        if (i.getBooleanExtra("bench", false)) {
            benchRuns = i.getIntExtra("runs", 3)
            if (benchButton.isEnabled) benchButton.performClick()
            return
        }
        if (i.hasExtra("text") && button.isEnabled) button.performClick()
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        runFromIntent(intent)
    }

    /** Save the last output as a 24 kHz mono 16-bit WAV in filesDir (adb-pullable). */
    private fun saveWav(audio: FloatArray, voice: String) {
        Wav.write(File(filesDir, "output.wav"), audio)
        Wav.write(File(filesDir, "output_$voice.wav"), audio)
    }

    private fun play(audio: FloatArray) {
        if (audio.isEmpty()) return
        val track = AudioTrack(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
            AudioFormat.Builder()
                .setSampleRate(PocketTtsSynthesizer.SAMPLE_RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
            audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
        track.play()
        Thread.sleep((audio.size * 1000L / PocketTtsSynthesizer.SAMPLE_RATE) + 250)
        track.release()
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdownNow()
        synth?.close()
    }
}
