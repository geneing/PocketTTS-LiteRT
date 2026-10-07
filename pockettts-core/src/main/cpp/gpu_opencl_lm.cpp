// Opt-in FlowLM GPU runner using only the public LiteRT 2.2.0 C ABI.
// LiteRT model buffers and the cache-row copy use public C/OpenCL interfaces.
#include "gpu_litert_c_api_220.h"
#include "gpu_opencl_public.h"

#include <jni.h>
#include <dlfcn.h>
#include <time.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
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
  GetEnvironmentOptions get_env_options = nullptr;
  GetEnvironmentOptionsValue get_env_value = nullptr;
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
  GetOpenClMemory opencl_memory = nullptr;
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
    RESOLVE(get_env_options, "LiteRtGetEnvironmentOptions");
    RESOLVE(get_env_value, "LiteRtGetEnvironmentOptionsValue");
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
    RESOLVE(opencl_memory, "LiteRtGetTensorBufferOpenClMemory");
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

struct ClApi {
  void* library = nullptr;
  opencl::GetMemObjectInfo get_mem_info = nullptr;
  opencl::GetContextInfo get_context_info = nullptr;
  opencl::GetCommandQueueInfo get_queue_info = nullptr;
  opencl::CreateCommandQueue create_queue = nullptr;
  opencl::ReleaseCommandQueue release_queue = nullptr;
  opencl::EnqueueReadBuffer read = nullptr;
  opencl::EnqueueCopyBuffer copy = nullptr;
  opencl::Finish finish = nullptr;
};

const ClApi& cl_api() {
  static const ClApi cl = [] {
    ClApi x;
    x.library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!x.library) throw std::runtime_error("dlopen libOpenCL.so failed: " +
                                              std::string(dlerror()));
#define RESOLVE(member, name)                                                  \
    x.member = reinterpret_cast<decltype(x.member)>(dlsym(x.library, name));  \
    if (!x.member) throw std::runtime_error(std::string("missing OpenCL symbol ") + name)
    RESOLVE(get_mem_info, "clGetMemObjectInfo");
    RESOLVE(get_context_info, "clGetContextInfo");
    RESOLVE(get_queue_info, "clGetCommandQueueInfo");
    RESOLVE(create_queue, "clCreateCommandQueue");
    RESOLVE(release_queue, "clReleaseCommandQueue");
    RESOLVE(read, "clEnqueueReadBuffer");
    RESOLVE(copy, "clEnqueueCopyBuffer");
    RESOLVE(finish, "clFinish");
#undef RESOLVE
    return x;
  }();
  return cl;
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
  const ClApi& cl = cl_api();
  opencl::Queue queue = nullptr;  // borrowed from LiteRT environment
  opencl::Mem cache_k = nullptr, cache_v = nullptr, output_mem = nullptr;
  bool cache_types_ok = false;
  std::string details;

  ~Runner() {
    if (output) a.destroy_buffer(output);
    for (auto& input : inputs) if (input) a.destroy_buffer(input);
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

  std::string requirement(Handle req, const char* label, bool* packed) {
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
      if (type == kOpenClBufferPacked && packed) *packed = true;
    }
    return s;
  }

  void init(const char* path) {
    checked(a.create_env(0, nullptr, &env), "create environment");
    checked(a.create_model(env, path, &model), "create model");
    checked(a.create_options(&options), "create options");
    checked(a.set_accelerators(options, kGpu), "request GPU");
    checked(a.create_compiled(env, model, options, &compiled), "compile GPU model");
    details = "LiteRT 2.2.0 C GPU OpenCL packed cache";

    std::array<Handle, kInputs> requirements{};
    for (int i = 0; i < kInputs; ++i) {
      checked(a.input_req(compiled, 0, i, &requirements[i]), "input requirements");
      bool opencl = false;
      details += "\n" + requirement(requirements[i],
                                     ("input" + std::to_string(i)).c_str(), &opencl);
      if (i == 4) cache_types_ok = opencl;
      if (i == 5) cache_types_ok = cache_types_ok && opencl;
      if (i == 4 || i == 5) {
        const Layout layout = input_type(i).layout;
        if (layout.rank != 4 || layout.has_strides ||
            layout.dimensions[0] != 1 || layout.dimensions[1] != kCapacity ||
            layout.dimensions[2] != kGroups || layout.dimensions[3] != kHeadDim) {
          throw std::runtime_error("cache input shape differs from [1,512,96,64]");
        }
        size_t bytes = 0;
        checked(a.req_size(requirements[i], &bytes), "cache requirement size");
        if (bytes != kCacheBytes) throw std::runtime_error("cache requirement size changed");
      }
    }
    Handle output_req = nullptr;
    checked(a.output_req(compiled, 0, 0, &output_req), "output requirements");
    details += "\n" + requirement(output_req, "output0", nullptr);
    size_t output_req_bytes = 0;
    checked(a.req_size(output_req, &output_req_bytes), "output requirement size");
    if (output_req_bytes != kOutput * sizeof(float)) {
      throw std::runtime_error("packed output requirement size changed");
    }
    Layout output_layout{};
    checked(a.output_layouts(compiled, 0, 1, &output_layout, false), "output layout");
    if (output_layout.rank != 2 || output_layout.has_strides ||
        output_layout.dimensions[0] != 1 || output_layout.dimensions[1] != kOutput) {
      throw std::runtime_error("packed output shape differs from [1,12321]");
    }
    details += std::string("\ncacheOpenClPackedSupported=") +
               (cache_types_ok ? "true" : "false");
    if (!cache_types_ok) return;  // Requirements gate is reportable.

    for (int i = 0; i < kInputs; ++i) {
      const RankedType type = input_type(i);
      checked(a.managed_from_req(env, &type, requirements[i], &inputs[i]),
              "create managed input buffer");
      int actual_type = -1;
      checked(a.buffer_type(inputs[i], &actual_type), "input buffer type");
      details += "\ninput" + std::to_string(i) + " actualType=" +
                 std::to_string(actual_type);
      if ((i == 4 || i == 5) && actual_type != kOpenClBufferPacked) {
        throw std::runtime_error("cache input not OpenCL packed");
      }
    }
    const RankedType output_type{kFloat32, output_layout};
    checked(a.managed_from_req(env, &output_type, output_req, &output),
            "create output buffer");
    int output_type_id = -1;
    checked(a.buffer_type(output, &output_type_id), "output buffer type");
    details += "\noutput0 actualType=" + std::to_string(output_type_id);
    if (output_type_id != kOpenClBufferPacked) {
      throw std::runtime_error("output not OpenCL packed");
    }
    checked(a.opencl_memory(inputs[4], &cache_k), "get cache K cl_mem");
    checked(a.opencl_memory(inputs[5], &cache_v), "get cache V cl_mem");
    checked(a.opencl_memory(output, &output_mem), "get output cl_mem");
    for (const auto [name, buffer, expected] : {
             std::tuple<const char*, opencl::Mem, size_t>{"cacheK", cache_k, kCacheBytes},
             {"cacheV", cache_v, kCacheBytes},
             // Pixel 10 OpenCL rounds the 49,284-byte packed output to 49,296.
             {"output", output_mem, 49296}}) {
      size_t actual = 0;
      checked(cl.get_mem_info(buffer, opencl::kMemSize, sizeof(actual),
                              &actual, nullptr), "clGetMemObjectInfo size");
      details += "\n" + std::string(name) + " clMemBytes=" + std::to_string(actual);
      if (actual != expected) throw std::runtime_error(std::string(name) + " cl_mem size changed");
    }
    opencl::Context context = nullptr;
    checked(cl.get_mem_info(cache_k, opencl::kMemContext, sizeof(context),
                            &context, nullptr), "clGetMemObjectInfo context");
    Handle env_options = nullptr;
    checked(a.get_env_options(env, &env_options), "get environment options");
    Any option{};
    const Status queue_status = a.get_env_value(env_options, 5, &option);
    if (queue_status != kOk || option.type != 2 || option.int_value == 0) {
      details += "\nLiteRT OpenCL command queue unavailable status=" +
                 std::to_string(queue_status) + " anyType=" + std::to_string(option.type);
      return;
    }
    queue = reinterpret_cast<opencl::Queue>(
        static_cast<intptr_t>(option.int_value));
    opencl::Context queue_context = nullptr;
    checked(cl.get_queue_info(queue, opencl::kQueueContext, sizeof(queue_context),
                              &queue_context, nullptr), "clGetCommandQueueInfo context");
    if (queue_context != context) {
      throw std::runtime_error("LiteRT queue differs from cache cl_mem context");
    }
    details += "\nclMem=true queue=borrowedLiteRT sameContext=true";
  }

  bool ready() const { return cache_types_ok && queue && cache_k && cache_v && output_mem; }

  void seed(JNIEnv* jni, jfloatArray k, jfloatArray v) {
    if (!ready()) throw std::runtime_error("OpenCL packed cache gate did not pass");
    const jsize n = kCapacity * kRow;
    if (jni->GetArrayLength(k) != n || jni->GetArrayLength(v) != n) {
      throw std::runtime_error("voice cache length differs from 512*96*64");
    }
    // The voice asset is group-major; the graph and OpenCL cache are position-major.
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
                   int position, bool full_output, jlongArray timings) {
    if (!ready()) throw std::runtime_error("OpenCL packed cache gate did not pass");
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
    std::array<float, kOutput> values{};
    // Run may return before the delegate finishes. This blocking read is
    // enqueued on LiteRT's own in-order OpenCL queue, after its graph kernels;
    // it waits for the result without mapping the whole 49 KB output.
    const size_t read_bytes = (full_output ? kOutput : kControl) * sizeof(float);
    checked(cl.read(queue, output_mem, opencl::kTrue, 0, read_bytes,
                    values.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer GPU output");
    const int64_t t3 = now_ns();
    const int64_t copy_start = now_ns();
    const size_t row_bytes = kRow * sizeof(float);
    checked(cl.copy(queue, output_mem, cache_k, kControl * sizeof(float),
                    static_cast<size_t>(position) * row_bytes, row_bytes,
                    0, nullptr, nullptr), "clEnqueueCopyBuffer K row");
    checked(cl.copy(queue, output_mem, cache_v, (kControl + kRow) * sizeof(float),
                    static_cast<size_t>(position) * row_bytes, row_bytes,
                    0, nullptr, nullptr), "clEnqueueCopyBuffer V row");
    const int64_t copy_end = now_ns();
    checked(cl.finish(queue), "clFinish cache row copies");
    const int64_t t4 = now_ns();
    const jlong measures[6] = {t1 - t0, t2 - t1, t3 - t2,
                               copy_start - t3, copy_end - copy_start,
                               t4 - copy_end};
    jni->SetLongArrayRegion(timings, 0, 6, measures);
    const int width = full_output ? kOutput : kControl;
    jfloatArray result = jni->NewFloatArray(width);
    if (result) jni->SetFloatArrayRegion(result, 0, width, values.data());
    return result;
  }
};

Runner* runner(jlong pointer) {
  if (!pointer) throw std::runtime_error("closed GPU OpenCL runner");
  return reinterpret_cast<Runner*>(static_cast<intptr_t>(pointer));
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_dev_pockettts_GpuOpenClLmBridge_open(JNIEnv* env, jobject, jstring path) {
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
Java_dev_pockettts_GpuOpenClLmBridge_details(JNIEnv* env, jobject, jlong pointer) {
  try {
    return env->NewStringUTF(runner(pointer)->details.c_str());
  } catch (const std::exception& e) {
    java_fail(env, e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_dev_pockettts_GpuOpenClLmBridge_ready(JNIEnv* env, jobject, jlong pointer) {
  try {
    return runner(pointer)->ready();
  } catch (const std::exception& e) {
    java_fail(env, e.what());
    return JNI_FALSE;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_dev_pockettts_GpuOpenClLmBridge_seed(JNIEnv* env, jobject, jlong pointer,
                                          jfloatArray k, jfloatArray v) {
  try {
    runner(pointer)->seed(env, k, v);
  } catch (const std::exception& e) {
    java_fail(env, e.what());
  }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_dev_pockettts_GpuOpenClLmBridge_step(JNIEnv* env, jobject, jlong pointer,
                                          jfloatArray emb, jfloatArray cos,
                                          jfloatArray sin, jfloatArray mask,
                                          jfloatArray noise, jint position,
                                          jboolean full_output, jlongArray timings) {
  try {
    return runner(pointer)->step(env, emb, cos, sin, mask, noise, position,
                                 full_output, timings);
  } catch (const std::exception& e) {
    java_fail(env, e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_dev_pockettts_GpuOpenClLmBridge_close(JNIEnv*, jobject, jlong pointer) {
  delete reinterpret_cast<Runner*>(static_cast<intptr_t>(pointer));
}

