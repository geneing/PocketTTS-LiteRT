plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.pockettts"
    compileSdk = 35

    defaultConfig {
        // 31+ because the Tensor G5 NPU dispatch runtime requires it.
        minSdk = 31
        consumerProguardFiles("consumer-rules.pro")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging {
        jniLibs {
            pickFirsts += setOf(
                "**/libc++_shared.so",
                "**/libtensorflowlite_jni.so",
                "**/libtensorflowlite_gpu_jni.so",
            )
        }
    }
}

dependencies {
    // Exposed transitively: every consumer needs the same 2.2.0 runtime the
    // Google Tensor dispatch shim and AOT compiler are pinned to.
    api("com.google.ai.edge.litert:litert:2.2.0")
}
