plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.pockettts"
    compileSdk = 35

    buildFeatures {
        buildConfig = true
    }

    defaultConfig {
        // 31+ because the Tensor G5 NPU dispatch runtime requires it.
        minSdk = 31
        consumerProguardFiles("consumer-rules.pro")
        externalNativeBuild {
            cmake { cppFlags += "-std=c++17" }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        // LiteRT 2.3.0's litert-api is built with Kotlin 2.4 metadata.
        freeCompilerArgs += "-Xskip-metadata-version-check"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    testOptions {
        // android.util.Log (used for the PocketTTSTime timing traces) is a stub
        // in local unit tests; return defaults instead of throwing.
        unitTests.isReturnDefaultValues = true
    }
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
    // Exposed transitively: every consumer needs the same 2.3.x runtime the
    // Google Tensor dispatch shim is pinned to. 2.3.0 split the GPU and NPU
    // runtimes out of the core AAR; 2.3.2 is the first dispatch shim that
    // initializes on Pixel 11 / Tensor G6.
    api("com.google.ai.edge.litert:litert:2.3.2")
    api("com.google.ai.edge.litert:litert-gpu:2.3.2")
    api("com.google.ai.edge.litert:litert-npu-runtime-google-tensor:2.3.2")

    testImplementation("junit:junit:4.13.2")
}
