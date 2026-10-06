// Experimental LiteRT 2.2.0 C-ABI path. The shipped Kotlin Options do not
// expose Google Tensor options; this supplies the documented opaque TOML payload.
#include <jni.h>
#include <dlfcn.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {
using Handle = void*;
using CreateModel = int (*)(Handle, const char*, Handle*);
using Destroy = void (*)(Handle);
using CreateOptions = int (*)(Handle*);
using SetAccelerator = int (*)(Handle, int);
using CreateOpaque = int (*)(const char*, void*, void (*)(void*), Handle*);
using AddOpaque = int (*)(Handle, Handle);
using FindOpaque = int (*)(Handle, const char*, void**);
using CreateCompiled = int (*)(Handle, Handle, Handle, Handle*);
using Run = int (*)(Handle, size_t, size_t, Handle*, size_t, Handle*);
using StatusString = const char* (*)(int);

struct Api {
  void* library = nullptr;
  CreateModel create_model = nullptr;
  Destroy destroy_model = nullptr;
  CreateOptions create_options = nullptr;
  Destroy destroy_options = nullptr;
  SetAccelerator set_accelerator = nullptr;
  CreateOpaque create_opaque = nullptr;
  Destroy destroy_opaque = nullptr;
  AddOpaque add_opaque = nullptr;
  FindOpaque find_opaque = nullptr;
  CreateCompiled create_compiled = nullptr;
  Destroy destroy_compiled = nullptr;
  Run run = nullptr;
  StatusString status_string = nullptr;
  bool valid() const {
    return library && create_model && destroy_model && create_options &&
           destroy_options && set_accelerator && create_opaque &&
           destroy_opaque && add_opaque && find_opaque && create_compiled &&
           destroy_compiled && run && status_string;
  }
};

const Api& api() {
  static const Api a = [] {
    Api x;
    x.library = dlopen("libLiteRt.so", RTLD_NOW | RTLD_LOCAL);
    if (!x.library) return x;
#define LOAD(member, symbol) \
    x.member = reinterpret_cast<decltype(x.member)>(dlsym(x.library, symbol))
    LOAD(create_model, "LiteRtCreateModelFromFile");
    LOAD(destroy_model, "LiteRtDestroyModel");
    LOAD(create_options, "LiteRtCreateOptions");
    LOAD(destroy_options, "LiteRtDestroyOptions");
    LOAD(set_accelerator, "LiteRtSetOptionsHardwareAccelerators");
    LOAD(create_opaque, "LiteRtCreateOpaqueOptions");
    LOAD(destroy_opaque, "LiteRtDestroyOpaqueOptions");
    LOAD(add_opaque, "LiteRtAddOpaqueOptions");
    LOAD(find_opaque, "LiteRtFindOpaqueOptionsData");
    LOAD(create_compiled, "LiteRtCreateCompiledModel");
    LOAD(destroy_compiled, "LiteRtDestroyCompiledModel");
    LOAD(run, "LiteRtRunCompiledModel");
    LOAD(status_string, "LiteRtGetStatusString");
#undef LOAD
    return x;
  }();
  return a;
}

void fail(JNIEnv* env, const std::string& message) {
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls) env->ThrowNew(cls, message.c_str());
}

bool ok(JNIEnv* env, int status, const char* operation) {
  if (status == 0) return true;
  const char* detail = api().status_string(status);
  fail(env, std::string(operation) + " failed (status " +
                std::to_string(status) + ", " + (detail ? detail : "unknown") + ")");
  return false;
}

// Kotlin JniHandle points to a C++ BaseHandle wrapper. The pinned 2.2.0
// wrapper's first word is the underlying C handle (also used by the KV bridge).
Handle c_handle(JNIEnv* env, jobject object) {
  if (!object) { fail(env, "null LiteRT JNI handle"); return nullptr; }
  jclass cls = env->FindClass("com/google/ai/edge/litert/JniHandle");
  if (!cls) return nullptr;
  jfieldID field = env->GetFieldID(cls, "handle", "J");
  if (!field) return nullptr;
  const jlong wrapper = env->GetLongField(object, field);
  if (!wrapper) { fail(env, "closed LiteRT JNI handle"); return nullptr; }
  static const bool compatible = [] {
    void* sentinel = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234));
    std::unique_ptr<void, std::function<void(void*)>> probe(sentinel, [](void*) {});
    void* first = nullptr;
    std::memcpy(&first, &probe, sizeof(first));
    return first == sentinel;
  }();
  if (!compatible) { fail(env, "LiteRT 2.2.0 C++ handle layout mismatch"); return nullptr; }
  Handle result = nullptr;
  std::memcpy(&result, reinterpret_cast<const void*>(static_cast<intptr_t>(wrapper)),
              sizeof(result));
  if (!result) fail(env, "null LiteRT C handle");
  return result;
}

// Unlike TensorBuffer, litert::Environment does not derive from BaseHandle.
// In v2.2.0 it stores runtime_ first, then handle_ (both unique_ptr objects
// with std::function deleters). Read the first word of handle_, not runtime_.
Handle environment_handle(JNIEnv* env, jobject object) {
  if (!object) { fail(env, "null LiteRT Environment"); return nullptr; }
  jclass cls = env->FindClass("com/google/ai/edge/litert/JniHandle");
  if (!cls) return nullptr;
  jfieldID field = env->GetFieldID(cls, "handle", "J");
  if (!field) return nullptr;
  const jlong wrapper = env->GetLongField(object, field);
  if (!wrapper) { fail(env, "closed LiteRT Environment"); return nullptr; }
  using Pointer = std::unique_ptr<void, std::function<void(void*)>>;
  Handle result = nullptr;
  std::memcpy(&result, reinterpret_cast<const char*>(static_cast<intptr_t>(wrapper)) +
                           sizeof(Pointer), sizeof(result));
  if (!result) fail(env, "null LiteRT C environment handle");
  return result;
}

struct NativeModel { Handle model = nullptr; Handle compiled = nullptr; };
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_dev_pockettts_G5PerformanceModel_00024Companion_nativeCreate(
    JNIEnv* env, jobject, jobject environment, jstring path) {
  const auto& a = api();
  if (!a.valid()) { fail(env, "LiteRT 2.2.0 C model API unavailable"); return 0; }
  if (!path) { fail(env, "null model path"); return 0; }
  Handle c_env = environment_handle(env, environment);
  if (env->ExceptionCheck()) return 0;
  const char* file = env->GetStringUTFChars(path, nullptr);
  if (!file) return 0;
  auto native = std::make_unique<NativeModel>();
  const int load_status = a.create_model(c_env, file, &native->model);
  env->ReleaseStringUTFChars(path, file);
  if (!ok(env, load_status, "LiteRtCreateModelFromFile")) return 0;
  Handle options = nullptr;
  if (!ok(env, a.create_options(&options), "LiteRtCreateOptions")) {
    a.destroy_model(native->model); return 0;
  }
  if (!ok(env, a.set_accelerator(options, 4), "LiteRtSetOptionsHardwareAccelerators(NPU)")) {
    a.destroy_options(options); a.destroy_model(native->model); return 0;
  }
  constexpr char kPayload[] = "performance_mode = 3\n";
  char* payload = static_cast<char*>(std::malloc(sizeof(kPayload)));
  if (!payload) {
    fail(env, "allocating Google Tensor TOML failed");
    a.destroy_options(options); a.destroy_model(native->model); return 0;
  }
  std::memcpy(payload, kPayload, sizeof(kPayload));
  Handle opaque = nullptr;
  if (!ok(env, a.create_opaque("google_tensor", payload, std::free, &opaque),
          "LiteRtCreateOpaqueOptions(google_tensor)")) {
    std::free(payload); a.destroy_options(options); a.destroy_model(native->model); return 0;
  }
  void* roundtrip = nullptr;
  if (!ok(env, a.find_opaque(opaque, "google_tensor", &roundtrip),
          "LiteRtFindOpaqueOptionsData") ||
      roundtrip != payload || std::strcmp(static_cast<const char*>(roundtrip), kPayload)) {
    if (!env->ExceptionCheck()) fail(env, "Google Tensor payload roundtrip mismatch");
    a.destroy_opaque(opaque); a.destroy_options(options); a.destroy_model(native->model);
    return 0;
  }
  if (!ok(env, a.add_opaque(options, opaque), "LiteRtAddOpaqueOptions")) {
    a.destroy_opaque(opaque); a.destroy_options(options); a.destroy_model(native->model);
    return 0;
  }
  const int compile_status = a.create_compiled(c_env, native->model, options,
                                               &native->compiled);
  a.destroy_options(options);
  if (!ok(env, compile_status, "LiteRtCreateCompiledModel(HIGH_PERFORMANCE)")) {
    a.destroy_model(native->model); return 0;
  }
  return reinterpret_cast<jlong>(native.release());
}

extern "C" JNIEXPORT void JNICALL
Java_dev_pockettts_G5PerformanceModel_nativeRun(
    JNIEnv* env, jobject, jlong native_handle, jobjectArray inputs,
    jobjectArray outputs, jint signature_index) {
  if (!native_handle || !inputs || !outputs || signature_index < 0) {
    fail(env, "invalid G5 performance model run arguments"); return;
  }
  std::vector<Handle> in(env->GetArrayLength(inputs));
  std::vector<Handle> out(env->GetArrayLength(outputs));
  for (size_t i = 0; i < in.size(); ++i) {
    jobject item = env->GetObjectArrayElement(inputs, i);
    in[i] = c_handle(env, item);
    env->DeleteLocalRef(item);
    if (env->ExceptionCheck()) return;
  }
  for (size_t i = 0; i < out.size(); ++i) {
    jobject item = env->GetObjectArrayElement(outputs, i);
    out[i] = c_handle(env, item);
    env->DeleteLocalRef(item);
    if (env->ExceptionCheck()) return;
  }
  auto* native = reinterpret_cast<NativeModel*>(native_handle);
  ok(env, api().run(native->compiled, static_cast<size_t>(signature_index),
                    in.size(), in.data(), out.size(), out.data()),
     "LiteRtRunCompiledModel(HIGH_PERFORMANCE)");
}

extern "C" JNIEXPORT void JNICALL
Java_dev_pockettts_G5PerformanceModel_nativeClose(JNIEnv*, jobject,
                                                    jlong native_handle) {
  if (!native_handle) return;
  auto* native = reinterpret_cast<NativeModel*>(native_handle);
  api().destroy_compiled(native->compiled);
  api().destroy_model(native->model);
  delete native;
}
