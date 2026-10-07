// Minimal declarations copied from the public LiteRT v2.2.0 C headers.
// Source: https://github.com/google-ai-edge/LiteRT/tree/v2.2.0/litert/c
// The Android AAR includes libLiteRt.so but does not package these headers.
// Resolve the named public C symbols at runtime from that same AAR library.
#pragma once

#include <android/hardware_buffer.h>
#include <cstddef>
#include <cstdint>

namespace litert220 {
using Handle = void*;
using Status = int;
constexpr Status kOk = 0;
constexpr int kGpu = 1 << 1;
constexpr int kFloat32 = 1;
constexpr int kAhwb = 2;
constexpr int kOpenClBufferPacked = 14;
constexpr int kRead = 0;
constexpr int kWrite = 1;

// litert_layout.h / litert_model_types.h: public ABI-stable layouts.
struct Layout {
  unsigned int rank : 7;
  unsigned int has_strides : 1;
  int32_t dimensions[8];
  uint32_t strides[8];
};
struct RankedType { int element_type; Layout layout; };
struct Any {
  int type;
  union { bool bool_value; int64_t int_value; double real_value;
          const char* str_value; const void* ptr_value; };
};
static_assert(sizeof(void*) != 8 || sizeof(Any) == 16);
static_assert(sizeof(Layout) == 68);
static_assert(offsetof(Layout, dimensions) == 4);
static_assert(offsetof(Layout, strides) == 36);
static_assert(sizeof(RankedType) == 72);
static_assert(offsetof(RankedType, layout) == 4);

using CreateEnvironment = Status (*)(int, const void*, Handle*);
using DestroyEnvironment = void (*)(Handle);
using GetEnvironmentOptions = Status (*)(Handle, Handle*);
using GetEnvironmentOptionsValue = Status (*)(Handle, int, Any*);
using SupportsAhwbClInterop = Status (*)(Handle, bool*);
using CreateModelFromFile = Status (*)(Handle, const char*, Handle*);
using DestroyModel = void (*)(Handle);
using CreateOptions = Status (*)(Handle*);
using DestroyOptions = void (*)(Handle);
using SetHardwareAccelerators = Status (*)(Handle, int);
using CreateCompiledModel = Status (*)(Handle, Handle, Handle, Handle*);
using DestroyCompiledModel = void (*)(Handle);
using GetInputRequirements = Status (*)(Handle, size_t, size_t, Handle*);
using GetOutputRequirements = Status (*)(Handle, size_t, size_t, Handle*);
using GetInputLayout = Status (*)(Handle, size_t, size_t, Layout*);
using GetOutputLayouts = Status (*)(Handle, size_t, size_t, Layout*, bool);
using GetNumSupportedTypes = Status (*)(Handle, int*);
using GetSupportedType = Status (*)(Handle, int, int*);
using GetRequirementSize = Status (*)(Handle, size_t*);
using GetRequirementAlignment = Status (*)(Handle, size_t*);
using CreateFromAhwb = Status (*)(Handle, const RankedType*, AHardwareBuffer*,
                                  size_t, void (*)(AHardwareBuffer*), Handle*);
using CreateManagedFromRequirements = Status (*)(Handle, const RankedType*,
                                                Handle, Handle*);
using GetOpenClMemory = Status (*)(Handle, void**);
using GetBufferType = Status (*)(Handle, int*);
using LockBuffer = Status (*)(Handle, void**, int);
using UnlockBuffer = Status (*)(Handle);
using DestroyBuffer = void (*)(Handle);
using RunModel = Status (*)(Handle, size_t, size_t, Handle*, size_t, Handle*);
}  // namespace litert220
