package dev.pockettts.service

import android.app.Activity
import android.os.Bundle
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import dev.pockettts.PocketTts

/**
 * The engine's settings screen, opened by the system TTS settings via the
 * `android:settingsActivity` meta-data on the service. Everything it edits is
 * stored with [PocketTtsSettings] and read back by [PocketTtsService] on the
 * next request, so the screen never has to talk to the running service.
 *
 * It exposes the three things the engine honours: which voice speaks a request
 * that does not name one, and the speech rate and pitch applied when a client
 * uses the framework defaults. Rate changes tempo (constant pitch) and pitch
 * changes tone (constant duration); both compose.
 */
class PocketTtsSettingsActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val voices = PocketTts.VOICES
        val pad = (resources.displayMetrics.density * 24).toInt()

        // Two nested rows: the outer one takes the action-bar inset, the inner
        // one owns the visual padding. (fitsSystemWindows on a padded view
        // replaces the padding instead of adding to it, so they must be split.)
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            fitsSystemWindows = true
            addView(content, matchWrap())
        }

        content.addView(header(getString(R.string.pockettts_settings_voice), top = 0, bottom = pad / 2))

        val current = PocketTtsSettings.voice(this, voices.first())
        val spinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@PocketTtsSettingsActivity,
                android.R.layout.simple_spinner_dropdown_item,
                voices,
            )
            setSelection(voices.indexOf(current).coerceAtLeast(0))
        }
        content.addView(spinner, matchWrap())

        // Speech rate: tempo only, pitch unchanged.
        content.addView(
            seekRow(
                title = getString(R.string.pockettts_settings_rate),
                value = PocketTtsSettings.rate(this),
                min = PocketTtsSettings.MIN_RATE,
                max = PocketTtsSettings.MAX_RATE,
                valueText = { getString(R.string.pockettts_settings_rate_value, it) },
                onChanged = { PocketTtsSettings.setRate(this, it) },
            ),
        )

        // Pitch: tone only, duration unchanged.
        content.addView(
            seekRow(
                title = getString(R.string.pockettts_settings_pitch),
                value = PocketTtsSettings.pitch(this),
                min = PocketTtsSettings.MIN_PITCH,
                max = PocketTtsSettings.MAX_PITCH,
                valueText = { getString(R.string.pockettts_settings_pitch_value, it) },
                onChanged = { PocketTtsSettings.setPitch(this, it) },
            ),
        )

        content.addView(TextView(this).apply {
            text = getString(R.string.pockettts_settings_note)
            textSize = 12f
            setPadding(0, pad, 0, 0)
        })

        spinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: android.widget.AdapterView<*>?,
                view: android.view.View?,
                position: Int,
                id: Long,
            ) {
                PocketTtsSettings.setVoice(this@PocketTtsSettingsActivity, voices[position])
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
        }

        setContentView(root)
        title = getString(R.string.pockettts_engine_name)
    }

    /** A titled label + SeekBar that writes each change straight to the prefs. */
    private fun seekRow(
        title: String,
        value: Float,
        min: Float,
        max: Float,
        valueText: (Float) -> String,
        onChanged: (Float) -> Unit,
    ): LinearLayout {
        val pad = (resources.displayMetrics.density * 24).toInt()
        val label = TextView(this).apply { textSize = 14f }
        val seek = SeekBar(this).apply {
            this.max = STEPS
            progress = toProgress(value, min, max)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    val v = fromProgress(p, min, max)
                    label.text = valueText(v)
                    onChanged(v)
                }

                override fun onStartTrackingTouch(sb: SeekBar?) = Unit
                override fun onStopTrackingTouch(sb: SeekBar?) = Unit
            })
        }
        label.text = valueText(value)
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(header(title, top = pad, bottom = pad / 2))
            addView(label, matchWrap())
            addView(seek, matchWrap())
        }
    }

    private fun header(title: String, top: Int, bottom: Int) = TextView(this).apply {
        text = title
        textSize = 16f
        setPadding(0, top, 0, bottom)
    }

    private fun matchWrap() = LinearLayout.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.WRAP_CONTENT,
    )

    private fun toProgress(value: Float, min: Float, max: Float): Int =
        ((value - min) / (max - min) * STEPS).toInt().coerceIn(0, STEPS)

    private fun fromProgress(progress: Int, min: Float, max: Float): Float =
        min + (max - min) * progress / STEPS.toFloat()

    private companion object {
        /** SeekBar is integer; 100 steps over a 4x range is ~4% resolution. */
        const val STEPS = 100
    }
}
