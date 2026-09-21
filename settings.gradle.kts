pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
dependencyResolutionManagement {
    repositories { google(); mavenCentral() }
}
rootProject.name = "PocketTTS"
include(":app")
include(":pockettts-core")
include(":pockettts-service")
