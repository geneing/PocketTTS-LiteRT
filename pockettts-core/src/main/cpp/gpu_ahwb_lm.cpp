// Opt-in FlowLM GPU runner using only the public LiteRT 2.2.0 C ABI.
// Kotlin's TensorBuffer does not expose an AHWB constructor in this release.
#include "gpu_litert_c_api_220.h"

#include <jni.h>
#include <dlfcn.h>
#include <time.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace litert220;
constexpr int kInputs = 7;
constexpr int kGroups = 96;
constexpr int kCapacity = 512;
constexpr int kHeadDim = 64;
constexpr int kControl = 33;
constexpr int kRow = kGroups * kHeadDim;
constexpr int kOutput = kControl + 2 * kRow;
constexpr size_t kCacheBytes = static_cast<size_t>(kCapacity) * kRow * sizeof(float);

int64_t now_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

void checked(Status status, const char* operation) {
  if (status != kOk) {
    throw std::runtime_error(std::string(operation) + " status=" +
                             std::to_string(status));
  }
}

void java_fail(JNIEnv* env, const std::string& message) {
  jclass type = env->FindClass("java/lang/IllegalStateException");
  if (type) env->ThrowNew(type, message.c_str());
}

struct Api {
  void* library = nullptr;
  CreateEnvironment create_env = nullptr;
  DestroyEnvironment destroy_env = nullptr;
  SupportsAhwbClInterop supports_ahwb_cl = nullptr;
  CreateModelFromFile create_model = nullptr;
  DestroyModel destroy_model = nullptr;
  CreateOptions create_options = nullptr;
  DestroyOptions destroy_options = nullptr;
  SetHardwareAccelerators set_accelerators = nullptr;
  CreateCompiledModel create_compiled = nullptr;
  DestroyCompiledModel destroy_compiled = nullptr;
  GetInputRequirements input_req = nullptr;
  GetOutputRequirements output_req = nullptr;
  GetInputLayout input_layout = nullptr;
  GetOutputLayouts output_layouts = nullptr;
  GetNumSupportedTypes num_types = nullptr;
  GetSupportedType supported_type = nullptr;
  GetRequirementSize req_size = nullptr;
  GetRequirementAlignment req_alignment = nullptr;
  CreateFromAhwb from_ahwb = nullptr;
  CreateManagedFromRequirements managed_from_req = nullptr;
  GetBufferType buffer_type = nullptr;
  LockBuffer lock = nullptr;
  UnlockBuffer unlock = nullptr;
  DestroyBuffer destroy_buffer = nullptr;
  RunModel run = nullptr;
};

const Api& api() {
  static const Api a = [] {
    Api x;
    x.library = dlopen("libLiteRt.so", RTLD_NOW | RTLD_LOCAL);
    if (!x.library) throw std::runtime_error("dlopen libLiteRt.so failed");
#define RESOLVE(member, name)                                                       \
    x.member = reinterpret_cast<decltype(x.member)>(dlsym(x.library, name));       \
    if (!x.member) throw std::runtime_error(std::string("missing public symbol ") + name)
    RESOLVE(create_env, "LiteRtCreateEnvironment");
    RESOLVE(destroy_env, "LiteRtDestroyEnvironment");
    RESOLVE(supports_ahwb_cl, "LiteRtEnvironmentSupportsAhwbClInterop");
    RESOLVE(create_model, "LiteRtCreateModelFromFile");
    RESOLVE(destroy_model, "LiteRtDestroyModel");
    RESOLVE(create_options, "LiteRtCreateOptions");
    RESOLVE(destroy_options, "LiteRtDestroyOptions");
    RESOLVE(set_accelerators, "LiteRtSetOptionsHardwareAccelerators");
    RESOLVE(create_compiled, "LiteRtCreateCompiledModel");
    RESOLVE(destroy_compiled, "LiteRtDestroyCompiledModel");
    RESOLVE(input_req, "LiteRtGetCompiledModelInputBufferRequirements");
    RESOLVE(output_req, "LiteRtGetCompiledModelOutputBufferRequirements");
    RESOLVE(input_layout, "LiteRtGetCompiledModelInputTensorLayout");
    RESOLVE(output_layouts, "LiteRtGetCompiledModelOutputTensorLayouts");
    RESOLVE(num_types, "LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes");
    RESOLVE(supported_type, "LiteRtGetTensorBufferRequirementsSupportedTensorBufferType");
    RESOLVE(req_size, "LiteRtGetTensorBufferRequirementsBufferSize");
    RESOLVE(req_alignment, "LiteRtGetTensorBufferRequirementsAlignment");
    RESOLVE(from_ahwb, "LiteRtCreateTensorBufferFromAhwb");
    RESOLVE(managed_from_req, "LiteRtCreateManagedTensorBufferFromRequirements");
    RESOLVE(buffer_type, "LiteRtGetTensorBufferType");
    RESOLVE(lock, "LiteRtLockTensorBuffer");
    RESOLVE(unlock, "LiteRtUnlockTensorBuffer");
    RESOLVE(destroy_buffer, "LiteRtDestroyTensorBuffer");
    RESOLVE(run, "LiteRtRunCompiledModel");
#undef RESOLVE
    return x;
  }();
  return a;
}

size_t elements(const Layout& shape) {
  if (shape.rank < 1 || shape.rank > 8 || shape.has_strides) {
    throw std::runtime_error("unsupported dynamic or strided FlowLM tensor");
  }
  size_t n = 1;
  for (unsigned int i = 0; i < shape.rank; ++i) {
    if (shape.dimensions[i] <= 0) throw std::runtime_error("dynamic FlowLM dimension");
    n *= static_cast<size_t>(shape.dimensions[i]);
  }
  return n;
}

struct Runner {
  const Api& a = api();
  Handle env = nullptr, model = nullptr, options = nullptr, compiled = nullptr;
  std::array<Handle, kInputs> inputs{};
  Handle output = nullptr;
  AHardwareBuffer *cache_k = nullptr, *cache_v = nullptr;
  bool interop = false;
  bool cache_types_ok = false;
  std::string details;

  ~Runner() {
    if (output) a.destroy_buffer(output);
    for (auto& input : inputs) if (input) a.destroy_buffer(input);
    if (cache_k) AHardwareBuffer_release(cache_k);
    if (cache_v) AHardwareBuffer_release(cache_v);
    if (compiled) a.destroy_compiled(compiled);
    if (options) a.destroy_options(options);
    if (model) a.destroy_model(model);
    if (env) a.destroy_env(env);
  }

  RankedType input_type(int index) {
    Layout layout{};
    checked(a.input_layout(compiled, 0, index, &layout), "input layout");
    return RankedType{kFloat32, layout};
  }

  std::string requirement(Handle req, const char* label, bool* ahwb) {
    int count = 0;
    size_t bytes = 0, alignment = 0;
    checked(a.num_types(req, &count), "requirement type count");
    checked(a.req_size(req, &bytes), "requirement size");
    checked(a.req_alignment(req, &alignment), "requirement alignment");
    std::string s = std::string(label) + " bytes=" + std::to_string(bytes) +
                    " alignment=" + std::to_string(alignment) + " types=";
    for (int i = 0; i < count; ++i) {
      int type = -1;
      checked(a.supported_type(req, i, &type), "requirement supported type");
      if (i) s += ",";
      s += std::to_string(type);
      if (type == kAhwb && ahwb) *ahwb = true;
    }
    return s;
  }

  AHardwareBuffer* allocate_cache(size_t bytes) {
    AHardwareBuffer_Desc desc{};
    desc.width = static_cast<uint32_t>(bytes);
    desc.height = 1;
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                 AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                 AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    AHardwareBuffer* result = nullptr;
    const int status = AHardwareBuffer_allocate(&desc, &result);
    if (status != 0 || !result) {
      throw std::runtime_error("AHardwareBuffer_allocate failed: " +
                               std::to_string(status));
    }
    return result;
  }

  void init(const char* path) {
    checked(a.create_env(0, nullptr, &env), "create environment");
    checked(a.create_model(env, path, &model), "create model");
    checked(a.create_options(&options), "create options");
    checked(a.set_accelerators(options, kGpu), "request GPU");
    checked(a.create_compiled(env, model, options, &compiled), "compile GPU model");
    checked(a.supports_ahwb_cl(env, &interop), "AHWB/CL interop query");
    details = std::string("LiteRT 2.2.0 C GPU ahwbClInterop=") +
              (interop ? "true" : "false");

    std::array<Handle, kInputs> requirements{};
    for (int i = 0; i < kInputs; ++i) {
      checked(a.input_req(compiled, 0, i, &requirements[i]), "input requirements");
      bool ahwb = false;
      details += "\n" + requirement(requirements[i],
                                     ("input" + std::to_string(i)).c_str(), &ahwb);
      if (i == 4) cache_types_ok = ahwb;
      if (i == 5) cache_types_ok = cache_types_ok && ahwb;
      if (i == 4 || i == 5) {
        if (elements(input_type(i).layout) * sizeof(float) != kCacheBytes) {
          throw std::runtime_error("cache input shape differs from [1,512,96,64]");
        }
        size_t bytes = 0;
        checked(a.req_size(requirements[i], &bytes), "cache requirement size");
        if (bytes < kCacheBytes) throw std::runtime_error("cache requirement too small");
      }
    }
    Handle output_req = nullptr;
    checked(a.output_req(compiled, 0, 0, &output_req), "output requirements");
    details += "\n" + requirement(output_req, "output0", nullptr);
    Layout output_layout{};
    checked(a.output_layouts(compiled, 0, 1, &output_layout, false), "output layout");
    if (elements(output_layout) != kOutput) {
      throw std::runtime_error("packed output size differs from 12321 floats");
    }
    details += std::string("\ncacheAhwbSupported=") +
               (cache_types_ok ? "true" : "false");
    if (!interop || !cache_types_ok) return;  // Gate reports a concrete stop reason.

    for (int i = 0; i < kInputs; ++i) {
      const RankedType type = input_type(i);
      if (i == 4 || i == 5) {
        size_t bytes = 0;
        checked(a.req_size(requirements[i], &bytes), "cache allocation size");
        AHardwareBuffer* hw = allocate_cache(bytes);
        if (i == 4) cache_k = hw; else cache_v = hw;
        checked(a.from_ahwb(env, &type, hw, 0, nullptr, &inputs[i]),
                "wrap cache AHWB");
      } else {
        checked(a.managed_from_req(env, &type, requirements[i], &inputs[i]),
                "create small input buffer");
      }
      int actual_type = -1;
      checked(a.buffer_type(inputs[i], &actual_type), "input buffer type");
      details += "\ninput" + std::to_string(i) + " actualType=" +
                 std::to_string(actual_type);
      if ((i == 4 || i == 5) && actual_type != kAhwb) {
        throw std::runtime_error("cache input not AHWB after wrap");
      }
    }
    const RankedType output_type{kFloat32, output_layout};
    checked(a.managed_from_req(env, &output_type, output_req, &output),
            "create output buffer");
    int output_type_id = -1;
    checked(a.buffer_type(output, &output_type_id), "output buffer type");
    details += "\noutput0 actualType=" + std::to_string(output_type_id);
  }

  bool ready() const { return interop && cache_types_ok && inputs[4] && inputs[5]; }

  void seed(JNIEnv* jni, jfloatArray k, jfloatArray v) {
    if (!ready()) throw std::runtime_error("AHWB/CL interop gate did not pass");
    const jsize n = kCapacity * kRow;
    if (jni->GetArrayLength(k) != n || jni->GetArrayLength(v) != n) {
      throw std::runtime_error("voice cache length differs from 512*96*64");
    }
    // The voice asset is group-major; the graph and AHWB cache are position-major.
    jboolean k_copy = JNI_FALSE, v_copy = JNI_FALSE;
    jfloat* source_k = jni->GetFloatArrayElements(k, &k_copy);
    jfloat* source_v = jni->GetFloatArrayElements(v, &v_copy);
    if (!source_k || !source_v) {
      if (source_k) jni->ReleaseFloatArrayElements(k, source_k, JNI_ABORT);
      if (source_v) jni->ReleaseFloatArrayElements(v, source_v, JNI_ABORT);
      throw std::runtime_error("could not map voice float arrays");
    }
    try {
      for (int bank = 0; bank < 2; ++bank) {
        void* mapped = nullptr;
        checked(a.lock(inputs[4 + bank], &mapped, kWrite), "lock seed cache");
        const float* source = bank == 0 ? source_k : source_v;
        float* destination = static_cast<float*>(mapped);
        for (int position = 0; position < kCapacity; ++position) {
          for (int group = 0; group < kGroups; ++group) {
            std::memcpy(destination + (position * kGroups + group) * kHeadDim,
                        source + (group * kCapacity + position) * kHeadDim,
                        kHeadDim * sizeof(float));
          }
        }
        checked(a.unlock(inputs[4 + bank]), "unlock seed cache");
      }
    } catch (...) {
      jni->ReleaseFloatArrayElements(k, source_k, JNI_ABORT);
      jni->ReleaseFloatArrayElements(v, source_v, JNI_ABORT);
      throw;
    }
    jni->ReleaseFloatArrayElements(k, source_k, JNI_ABORT);
    jni->ReleaseFloatArrayElements(v, source_v, JNI_ABORT);
  }

  void write(JNIEnv* jni, int index, jfloatArray source) {
    const RankedType type = input_type(index);
    const size_t n = elements(type.layout);
    if (!source || jni->GetArrayLength(source) != static_cast<jsize>(n)) {
      throw std::runtime_error("small input" + std::to_string(index) + " length mismatch");
    }
    void* mapped = nullptr;
    checked(a.lock(inputs[index], &mapped, kWrite), "lock small input");
    jni->GetFloatArrayRegion(source, 0, static_cast<jsize>(n),
                             static_cast<jfloat*>(mapped));
    checked(a.unlock(inputs[index]), "unlock small input");
    if (jni->ExceptionCheck()) throw std::runtime_error("write small input failed");
  }

  jfloatArray step(JNIEnv* jni, jfloatArray emb, jfloatArray cos,
                   jfloatArray sin, jfloatArray mask, jfloatArray noise,
                   int position, jlongArray timings) {
    if (!ready()) throw std::runtime_error("AHWB/CL interop gate did not pass");
    if (position < 0 || position >= kCapacity || !timings ||
        jni->GetArrayLength(timings) < 6) {
      throw std::runtime_error("invalid cache position or timing array");
    }
    const int64_t t0 = now_ns();
    write(jni, 0, emb); write(jni, 1, cos); write(jni, 2, sin);
    write(jni, 3, mask); write(jni, 6, noise);
    const int64_t t1 = now_ns();
    checked(a.run(compiled, 0, kInputs, inputs.data(), 1, &output), "GPU step");
    const int64_t t2 = now_ns();
    void* mapped = nullptr;
    checked(a.lock(output, &mapped, kRead), "lock packed output");
    std::array<float, kOutput> values{};
    std::memcpy(values.data(), mapped, values.size() * sizeof(float));
    checked(a.unlock(output), "unlock packed output");
    const int64_t t3 = now_ns();
    const int64_t map_start = now_ns();
    void* target_k = nullptr;
    void* target_v = nullptr;
    checked(a.lock(inputs[4], &target_k, kWrite), "lock cache K row");
    checked(a.lock(inputs[5], &target_v, kWrite), "lock cache V row");
    const int64_t copy_start = now_ns();
    std::memcpy(static_cast<float*>(target_k) + position * kRow,
                values.data() + kControl, kRow * sizeof(float));
    std::memcpy(static_cast<float*>(target_v) + position * kRow,
                values.data() + kControl + kRow, kRow * sizeof(float));
    const int64_t copy_end = now_ns();
    checked(a.unlock(inputs[4]), "unlock cache K row");
    checked(a.unlock(inputs[5]), "unlock cache V row");
    const int64_t t4 = now_ns();
    const jlong measures[6] = {t1 - t0, t2 - t1, t3 - t2,
                               copy_start - map_start, copy_end - copy_start,
                               t4 - copy_end};
    jni->SetLongArrayRegion(timings, 0, 6, measures);
    jfloatArray result = jni->NewFloatArray(kOutput);
    if (result) jni->SetFloatArrayRegion(result, 0, kOutput, values.data());
    return result;
  }
};

Runner* runner(jlong pointer) {
  if (!pointer) throw std::runtime_error("closed GPU AHWB runner");
  return reinterpret_cast<Runner*>(static_cast<intptr_t>(pointer));
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_dev_pockettts_GpuAhwbLmBridge_open(JNIEnv* env, jobject, jstring path) {
  try {
    if (!path) throw std::runtime_error("null FlowLM graph path");
    const char* utf = env->GetStringUTFChars(path, nullptr);
    if (!utf) return 0;
    std::string name(utf);
    env->ReleaseStringUTFChars(path, utf);
    auto value = std::make_unique<Runner>();
    value->init(name.c_str());
    return static_cast<jlong>(reinterpret_cast<intptr_t>(value.release()));
  } catch (const std::exception& e) {
    java_fail(env, e.what());
    return 0;
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_pockettts_GpuAhwbLmBridge_details(JNIEnv* env, jobject, jlong pointer) {
  try {
    return env->NewStringUTF(runner(pointer)->details.c_str());
  } catch (const std::exception& e) {
    java_fail(env, e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_dev_pockettts_GpuAhwbLmBridge_ready(JNIEnv* env, jobject, jlong pointer) {
  try {
    return runner(pointer)->ready();
  } catch (const std::exception& e) {
    java_fail(env, e.what());
    return JNI_FALSE;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_dev_pockettts_GpuAhwbLmBridge_seed(JNIEnv* env, jobject, jlong pointer,
                                          jfloatArray k, jfloatArray v) {
  try {
    runner(pointer)->seed(env, k, v);
  } catch (const std::exception& e) {
    java_fail(env, e.what());
  }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_dev_pockettts_GpuAhwbLmBridge_step(JNIEnv* env, jobject, jlong pointer,
                                          jfloatArray emb, jfloatArray cos,
                                          jfloatArray sin, jfloatArray mask,
                                          jfloatArray noise, jint position,
                                          jlongArray timings) {
  try {
    return runner(pointer)->step(env, emb, cos, sin, mask, noise, position, timings);
  } catch (const std::exception& e) {
    java_fail(env, e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_dev_pockettts_GpuAhwbLmBridge_close(JNIEnv*, jobject, jlong pointer) {
  delete reinterpret_cast<Runner*>(static_cast<intptr_t>(pointer));
}
