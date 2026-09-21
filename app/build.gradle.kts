plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.pockettts"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.pockettts"
        // 31+ because the NPU dispatch runtime requires it; only arm64-v8a has
        // an NPU dispatch shim at all.
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += setOf("arm64-v8a") }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging {
        jniLibs {
            // The NPU dispatch shim is dlopen'ed by absolute path out of the
            // installed package's lib/arm64/ directory, so the .so has to be
            // extracted at install time rather than left compressed in the APK.
            // Without this the dispatch lookup finds nothing and the model never
            // reaches the NPU.
            useLegacyPackaging = true
            pickFirsts += setOf(
                "**/libc++_shared.so",
                "**/libtensorflowlite_jni.so",
                "**/libtensorflowlite_gpu_jni.so",
            )
        }
    }
}
dependencies {
    // The reusable engine (brings litert 2.2.0 transitively) and the TTS engine
    // service. 2.2.0: latest on Google Maven (2.1.6 pinned earlier for Mali;
    // 2.1.5+ accepts the KV-step FULLY_CONNECTED weight shapes).
    implementation(project(":pockettts-core"))
    implementation(project(":pockettts-service"))
    implementation("androidx.core:core-ktx:1.15.0")
}
