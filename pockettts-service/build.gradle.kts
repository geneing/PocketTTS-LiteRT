plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.pockettts.service"
    compileSdk = 35

    defaultConfig {
        minSdk = 31
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
}

dependencies {
    api(project(":pockettts-core"))
}
