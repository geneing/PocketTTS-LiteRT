// Minimal Khronos OpenCL 1.2 C declarations used by the opt-in GPU cache path.
// The Android NDK does not package CL headers. These calls are loaded by name.
#pragma once
#include <cstddef>
#include <cstdint>

namespace opencl {
using Context = void*;
using Queue = void*;
using Device = void*;
using Mem = void*;
using Int = int32_t;
using UInt = uint32_t;
using Bool = UInt;
constexpr Int kSuccess = 0;
constexpr Bool kTrue = 1;
constexpr UInt kMemContext = 0x1106;
constexpr UInt kMemSize = 0x1102;
constexpr UInt kContextDevices = 0x1081;
constexpr UInt kQueueContext = 0x1090;
using GetMemObjectInfo = Int (*)(Mem, UInt, size_t, void*, size_t*);
using GetContextInfo = Int (*)(Context, UInt, size_t, void*, size_t*);
using GetCommandQueueInfo = Int (*)(Queue, UInt, size_t, void*, size_t*);
using CreateCommandQueue = Queue (*)(Context, Device, uint64_t, Int*);
using ReleaseCommandQueue = Int (*)(Queue);
using EnqueueReadBuffer = Int (*)(Queue, Mem, Bool, size_t, size_t, void*, UInt,
                                  const void*, void*);
using EnqueueCopyBuffer = Int (*)(Queue, Mem, Mem, size_t, size_t, size_t, UInt,
                                  const void*, void*);
using Finish = Int (*)(Queue);
}  // namespace opencl
