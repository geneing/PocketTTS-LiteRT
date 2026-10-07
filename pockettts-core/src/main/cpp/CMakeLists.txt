cmake_minimum_required(VERSION 3.22.1)
project(pockettts_kv_slice LANGUAGES CXX)

add_library(pockettts_kv_slice SHARED npu_slice_cache.cpp)
target_link_libraries(pockettts_kv_slice PRIVATE dl log)

# Public LiteRT 2.2.0 C API, resolved from the app's packaged libLiteRt.so.
add_library(pockettts_gpu_ahwb SHARED gpu_ahwb_lm.cpp)
target_link_libraries(pockettts_gpu_ahwb PRIVATE android dl log)
