package dev.pockettts.service

import android.content.Context

/**
 * The TTS engine's user-visible settings, stored in a private
 * [android.content.SharedPreferences] file shared by [PocketTtsService] and
 * [PocketTtsSettingsActivity]. Kept tiny on purpose: a voice and a rate.
 *
 * The Android TTS framework owns the *default* locale and rate in
 * `Settings.Secure`, but engines cannot write those from the service process;
 * this is the engine-local preference the settings screen edits.
 */
object PocketTtsSettings {

    private const val PREFS = "pockettts_settings"
    private const val KEY_VOICE = "voice"
    private const val KEY_RATE = "rate"
    private const val KEY_PITCH = "pitch"

    /** ~0.5x to 2x; applied as a post-decode PCM time-stretch, not a graph change. */
    const val MIN_RATE = 0.5f
    const val MAX_RATE = 2.0f

    /** ~0.5x to 2x; a pitch shift that keeps the utterance duration. */
    const val MIN_PITCH = 0.5f
    const val MAX_PITCH = 2.0f

    private fun prefs(context: Context) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** The voice name to speak when a request does not name one. */
    fun voice(context: Context, fallback: String): String =
        prefs(context).getString(KEY_VOICE, null)
            ?: fallback

    fun setVoice(context: Context, name: String) {
        prefs(context).edit().putString(KEY_VOICE, name).apply()
    }

    /** Speech rate, 1.0 = natural. Requests with the framework default use it. */
    fun rate(context: Context): Float =
        prefs(context).getFloat(KEY_RATE, 1f).coerceIn(MIN_RATE, MAX_RATE)

    fun setRate(context: Context, rate: Float) {
        prefs(context).edit().putFloat(KEY_RATE, rate.coerceIn(MIN_RATE, MAX_RATE)).apply()
    }

    /** Voice pitch, 1.0 = natural. Requests with the framework default use it. */
    fun pitch(context: Context): Float =
        prefs(context).getFloat(KEY_PITCH, 1f).coerceIn(MIN_PITCH, MAX_PITCH)

    fun setPitch(context: Context, pitch: Float) {
        prefs(context).edit().putFloat(KEY_PITCH, pitch.coerceIn(MIN_PITCH, MAX_PITCH)).apply()
    }
}
