package com.pockettts

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.pockettts.PocketTtsModels
import dev.pockettts.VoiceCatalog
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Voice discovery: a cache pushed into the app's external files dir must show up
 * as a first-class voice, for the engine and the TTS service alike.
 *
 * Guards the regression where `Voice.all()` (the hard-coded bundled list) was the
 * only thing consulted, so `pt_voice_bob.bin` was installed but unusable.
 */
@RunWith(AndroidJUnit4::class)
class VoiceCatalogTest {

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun discoversPushedVoices() {
        val models = PocketTtsModels.default(ctx)
        // The app must own voices/, or adb-created files in it are unreadable.
        Log.i("VoiceCatalog", "ensureDir=${VoiceCatalog.ensureDir(models)}")
        val installed = VoiceCatalog.installed(models)
        val names = installed.map { it.name }
        Log.i("VoiceCatalog", "installed=$names")

        assertTrue("bundled voices present", names.containsAll(listOf("alba", "mary")))

        // Any pt_voice_*.bin in the app's external dir counts as a voice, whatever
        // it is called; assert the mechanism, not a specific generated cache.
        val custom = names.filterNot { it in BUNDLED }
        assertTrue("no custom voice discovered in $names", custom.isNotEmpty())

        for (name in custom) {
            assertTrue("$name must resolve by bare name", VoiceCatalog.named(installed, name)?.name == name)
            assertTrue(
                "$name must resolve by pockettts- id",
                VoiceCatalog.named(installed, "pockettts-$name")?.name == name,
            )
        }
        assertTrue("unknown name must not resolve", VoiceCatalog.named(installed, "nope") == null)
    }

    private companion object {
        val BUNDLED = listOf("alba", "marius", "javert", "charles", "mary", "eve")
    }
}
