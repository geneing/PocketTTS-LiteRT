// LiteRT 2.2.0 C ABI: litert/c/litert_tensor_buffer.h. Resolve from the AAR's
// libLiteRt.so so the bridge uses the exact runtime that owns Kotlin buffers.
#include <jni.h>
#include <dlfcn.h>
#include <time.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>

namespace {
using Buffer = void*;
using GetType = int (*)(Buffer, int*);
using GetPackedSize = int (*)(Buffer, size_t*);
using GetSize = int (*)(Buffer, size_t*);
// LiteRT 2.2.0 public, ABI-stable structs from litert_layout.h and
// litert_model_types.h. Keep these checks beside the dlsym-based bridge.
struct Layout {
  unsigned int rank : 7;
  unsigned int has_strides : 1;
  int32_t dimensions[8];
  uint32_t strides[8];
};
struct RankedType { int element_type; Layout layout; };
static_assert(sizeof(Layout) == 68);
static_assert(offsetof(Layout, dimensions) == 4);
static_assert(offsetof(Layout, strides) == 36);
static_assert(sizeof(RankedType) == 72);
static_assert(offsetof(RankedType, layout) == 4);
using GetTensorType = int (*)(Buffer, RankedType*);
using Lock = int (*)(Buffer, void**, int);
using Unlock = int (*)(Buffer);
constexpr int kOk = 0;
constexpr int kRead = 0;
constexpr int kReadWrite = 2;
constexpr int kControl = 33;

struct Api {
  void* library = nullptr;
  GetType get_type = nullptr;
  GetPackedSize packed_size = nullptr;
  GetSize size = nullptr;
  GetSize offset = nullptr;
  GetTensorType tensor_type = nullptr;
  Lock lock = nullptr;
  Unlock unlock = nullptr;
  bool valid() const {
    return library && get_type && packed_size && size && offset &&
           tensor_type && lock && unlock;
  }
};

const Api& api() {
  static const Api loaded = [] {
    Api a;
    a.library = dlopen("libLiteRt.so", RTLD_NOW | RTLD_LOCAL);
    if (!a.library) return a;
    a.get_type = reinterpret_cast<GetType>(dlsym(a.library, "LiteRtGetTensorBufferType"));
    a.packed_size = reinterpret_cast<GetPackedSize>(dlsym(a.library, "LiteRtGetTensorBufferPackedSize"));
    a.size = reinterpret_cast<GetSize>(dlsym(a.library, "LiteRtGetTensorBufferSize"));
    a.offset = reinterpret_cast<GetSize>(dlsym(a.library, "LiteRtGetTensorBufferOffset"));
    a.tensor_type = reinterpret_cast<GetTensorType>(dlsym(a.library, "LiteRtGetTensorBufferTensorType"));
    a.lock = reinterpret_cast<Lock>(dlsym(a.library, "LiteRtLockTensorBuffer"));
    a.unlock = reinterpret_cast<Unlock>(dlsym(a.library, "LiteRtUnlockTensorBuffer"));
    return a;
  }();
  return loaded;
}

void fail(JNIEnv* env, const std::string& message) {
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls) env->ThrowNew(cls, message.c_str());
}

int64_t now_ns() {
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<int64_t>(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

Buffer handle(JNIEnv* env, jobject tensor_buffer) {
  if (!tensor_buffer) {
    fail(env, "null LiteRT TensorBuffer");
    return nullptr;
  }
  // JniHandle.handle points to a C++ litert::TensorBuffer, NOT to the C
  // LiteRtTensorBuffer. In LiteRT 2.2.0 the wrapper has one nonvirtual
  // BaseHandle<LiteRtTensorBuffer> base, whose first member is a unique_ptr
  // holding the C handle (litert/cc/internal/litert_handle.h). The JNI in
  // litert_tensor_buffer_jni.cc casts this Java value to TensorBuffer*.
  // This private ABI bridge must be revisited on any LiteRT/NDK update.
  jclass base = env->FindClass("com/google/ai/edge/litert/JniHandle");
  if (!base) return nullptr;
  jfieldID field = env->GetFieldID(base, "handle", "J");
  if (!field) return nullptr;
  auto value = env->GetLongField(tensor_buffer, field);
  if (!value) { fail(env, "closed LiteRT TensorBuffer"); return nullptr; }
  static const bool first_word_is_unique_ptr = [] {
    // Check the NDK libc++ layout instead of assuming where unique_ptr keeps
    // its pointer when it has a std::function deleter.
    void* sentinel = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234));
    std::unique_ptr<void, std::function<void(void*)>> probe(
        sentinel, [](void*) {});
    void* first_word = nullptr;
    std::memcpy(&first_word, &probe, sizeof(first_word));
    return first_word == sentinel;
  }();
  if (!first_word_is_unique_ptr) {
    fail(env, "LiteRT C++ TensorBuffer wrapper layout differs from pinned ABI");
    return nullptr;
  }
  Buffer c_handle = nullptr;
  std::memcpy(&c_handle, reinterpret_cast<const void*>(
                             static_cast<intptr_t>(value)), sizeof(c_handle));
  if (!c_handle) fail(env, "LiteRT C++ TensorBuffer contains a null C handle");
  return c_handle;
}

bool expect_size(JNIEnv* env, const Api& a, Buffer buffer, size_t bytes,
                 const char* label) {
  size_t packed = 0, allocation = 0, offset = 0;
  int buffer_type = -1;
  RankedType tensor_type{};
  const int packed_status = a.packed_size(buffer, &packed);
  const int size_status = a.size(buffer, &allocation);
  const int offset_status = a.offset(buffer, &offset);
  const int type_status = a.get_type(buffer, &buffer_type);
  const int tensor_status = a.tensor_type(buffer, &tensor_type);
  size_t tensor_elements = 1;
  bool tensor_shape_valid = tensor_status == kOk &&
                            tensor_type.element_type == 1 &&
                            tensor_type.layout.rank >= 1 &&
                            tensor_type.layout.rank <= 8;
  if (tensor_shape_valid) {
    for (unsigned int i = 0; i < tensor_type.layout.rank; ++i) {
      const int32_t dim = tensor_type.layout.dimensions[i];
      if (dim <= 0 || tensor_elements > bytes / sizeof(float) /
                                          static_cast<size_t>(dim)) {
        tensor_shape_valid = false;
        break;
      }
      tensor_elements *= static_cast<size_t>(dim);
    }
  }
  if (packed_status != kOk || packed != bytes || size_status != kOk ||
      offset_status != kOk || type_status != kOk || tensor_status != kOk ||
      buffer_type <= 0 || tensor_shape_valid == false ||
      tensor_elements != bytes / sizeof(float) || offset > allocation ||
      allocation - offset < bytes) {
    std::string shape = "[";
    if (tensor_status == kOk && tensor_type.layout.rank <= 8) {
      for (unsigned int i = 0; i < tensor_type.layout.rank; ++i) {
        if (i) shape += ",";
        shape += std::to_string(tensor_type.layout.dimensions[i]);
      }
    } else {
      shape += "unknown";
    }
    shape += "]";
    fail(env, std::string(label) + " buffer mismatch: expectedPacked=" +
                  std::to_string(bytes) + " packed=" + std::to_string(packed) +
                  " allocation=" + std::to_string(allocation) +
                  " offset=" + std::to_string(offset) +
                  " bufferType=" + std::to_string(buffer_type) +
                  " elementType=" + std::to_string(tensor_type.element_type) +
                  " rank=" + std::to_string(tensor_type.layout.rank) +
                  " shape=" + shape + " hasStrides=" +
                  std::to_string(tensor_type.layout.has_strides) +
                  " shapeValid=" + std::to_string(tensor_shape_valid) +
                  " statuses=" + std::to_string(packed_status) + "," +
                  std::to_string(size_status) + "," +
                  std::to_string(offset_status) + "," +
                  std::to_string(type_status) + "," +
                  std::to_string(tensor_status));
    return false;
  }
  return true;
}

struct Mapped {
  const Api& a;
  Buffer buffer;
  void* data = nullptr;
  bool locked = false;
  Mapped(const Api& api, Buffer b) : a(api), buffer(b) {}
  ~Mapped() { if (locked) a.unlock(buffer); }
  bool map(JNIEnv* env, int mode, const char* label) {
    const int status = a.lock(buffer, &data, mode);
    if (status != kOk || !data) {
      fail(env, std::string("LiteRT lock ") + label + " failed: " +
                    std::to_string(status));
      return false;
    }
    locked = true;
    return true;
  }
  bool unmap(JNIEnv* env, const char* label) {
    if (!locked) return true;
    locked = false;
    const int status = a.unlock(buffer);
    if (status != kOk) {
      fail(env, std::string("LiteRT unlock ") + label + " failed: " +
                    std::to_string(status));
      return false;
    }
    return true;
  }
};

bool geometry(JNIEnv* env, int position, int capacity, int groups, int head_dim) {
  if (position < 0 || capacity <= 0 || position >= capacity ||
      groups <= 0 || head_dim <= 0) {
    fail(env, "invalid FlowLM cache geometry or position");
    return false;
  }
  return true;
}
}  // namespace

extern "C" JNIEXPORT jintArray JNICALL
Java_dev_pockettts_NpuSliceCacheBridge_bufferTypes(
    JNIEnv* env, jobject, jobject cache_k_obj, jobject cache_v_obj,
    jobject output_obj) {
  const auto& a = api();
  if (!a.valid()) { fail(env, "LiteRT 2.2.0 tensor buffer C ABI unavailable"); return nullptr; }
  Buffer buffers[] = {handle(env, cache_k_obj), handle(env, cache_v_obj),
                      handle(env, output_obj)};
  if (env->ExceptionCheck()) return nullptr;
  jint types[3]{};
  for (int i = 0; i < 3; ++i) {
    const int status = a.get_type(buffers[i], &types[i]);
    if (status != kOk) {
      fail(env, "LiteRT buffer type query failed: " + std::to_string(status));
      return nullptr;
    }
  }
  jintArray result = env->NewIntArray(3);
  if (result) env->SetIntArrayRegion(result, 0, 3, types);
  return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_dev_pockettts_NpuSliceCacheBridge_update(
    JNIEnv* env, jobject, jobject cache_k_obj, jobject cache_v_obj,
    jobject output_obj, jint position, jint capacity, jint groups,
    jint head_dim, jlongArray timings) {
  const auto& a = api();
  if (!a.valid()) { fail(env, "LiteRT 2.2.0 tensor buffer C ABI unavailable"); return nullptr; }
  if (!geometry(env, position, capacity, groups, head_dim)) return nullptr;
  if (!timings || env->GetArrayLength(timings) < 4) {
    fail(env, "cache timing array must have four entries"); return nullptr;
  }
  Buffer k = handle(env, cache_k_obj);
  Buffer v = handle(env, cache_v_obj);
  Buffer out = handle(env, output_obj);
  if (env->ExceptionCheck()) return nullptr;
  const size_t row_floats = static_cast<size_t>(groups) * head_dim;
  const size_t cache_floats = row_floats * capacity;
  if (!expect_size(env, a, k, cache_floats * sizeof(float), "K cache") ||
      !expect_size(env, a, v, cache_floats * sizeof(float), "V cache") ||
      !expect_size(env, a, out, (kControl + 2 * row_floats) * sizeof(float),
                   "FlowLM output")) return nullptr;

  Mapped output(a, out), cache_k(a, k), cache_v(a, v);
  const int64_t t0 = now_ns();
  if (!output.map(env, kRead, "FlowLM output")) return nullptr;
  const int64_t t1 = now_ns();
  if (!cache_k.map(env, kReadWrite, "K cache") ||
      !cache_v.map(env, kReadWrite, "V cache")) return nullptr;
  const int64_t t2 = now_ns();

  const auto* values = static_cast<const float*>(output.data);
  auto* keys = static_cast<float*>(cache_k.data);
  auto* vals = static_cast<float*>(cache_v.data);
  jfloatArray control = env->NewFloatArray(kControl);
  if (!control) return nullptr;
  env->SetFloatArrayRegion(control, 0, kControl, values);
  if (env->ExceptionCheck()) return nullptr;
  for (int g = 0; g < groups; ++g) {
    const size_t dst = (static_cast<size_t>(g) * capacity + position) * head_dim;
    const size_t src = static_cast<size_t>(g) * head_dim;
    std::memcpy(keys + dst, values + kControl + src, head_dim * sizeof(float));
    std::memcpy(vals + dst, values + kControl + row_floats + src,
                head_dim * sizeof(float));
  }
  const int64_t t3 = now_ns();
  if (!cache_k.unmap(env, "K cache") || !cache_v.unmap(env, "V cache") ||
      !output.unmap(env, "FlowLM output")) return nullptr;
  const int64_t t4 = now_ns();
  const jlong stages[] = {t1 - t0, t2 - t1, t3 - t2, t4 - t3};
  env->SetLongArrayRegion(timings, 0, 4, stages);
  return control;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_dev_pockettts_NpuSliceCacheBridge_rowMaxDifference(
    JNIEnv* env, jobject, jobject cache_k_obj, jobject cache_v_obj,
    jobject output_obj, jint position, jint capacity, jint groups,
    jint head_dim) {
  const auto& a = api();
  if (!a.valid()) { fail(env, "LiteRT 2.2.0 tensor buffer C ABI unavailable"); return 0; }
  if (!geometry(env, position, capacity, groups, head_dim)) return 0;
  Buffer k = handle(env, cache_k_obj);
  Buffer v = handle(env, cache_v_obj);
  Buffer out = handle(env, output_obj);
  if (env->ExceptionCheck()) return 0;
  const size_t row_floats = static_cast<size_t>(groups) * head_dim;
  const size_t cache_floats = row_floats * capacity;
  if (!expect_size(env, a, k, cache_floats * sizeof(float), "K cache") ||
      !expect_size(env, a, v, cache_floats * sizeof(float), "V cache") ||
      !expect_size(env, a, out, (kControl + 2 * row_floats) * sizeof(float),
                   "FlowLM output")) return 0;
  Mapped output(a, out), cache_k(a, k), cache_v(a, v);
  if (!output.map(env, kRead, "FlowLM output") ||
      !cache_k.map(env, kRead, "K cache") ||
      !cache_v.map(env, kRead, "V cache")) return 0;
  const auto* values = static_cast<const float*>(output.data);
  const auto* keys = static_cast<const float*>(cache_k.data);
  const auto* vals = static_cast<const float*>(cache_v.data);
  float max_delta = 0;
  for (int g = 0; g < groups; ++g) {
    const size_t dst = (static_cast<size_t>(g) * capacity + position) * head_dim;
    const size_t src = static_cast<size_t>(g) * head_dim;
    for (int h = 0; h < head_dim; ++h) {
      max_delta = std::max(max_delta, std::abs(keys[dst + h] - values[kControl + src + h]));
      max_delta = std::max(max_delta, std::abs(vals[dst + h] - values[kControl + row_floats + src + h]));
    }
  }
  return max_delta;
}
