# FLUX.2-klein-4B ExecuTorch Runner
#
# Builds against an ExecuTorch install tree that has XNNPACK + optimized/quantized
# kernels + (optionally) QNN backends compiled in.
#
# Build for Android ARM64:
#   mkdir build-android && cd build-android
#   cmake .. \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_PLATFORM=android-30 \
#     -DCMAKE_BUILD_TYPE=Release \
#     -Dexecutorch_DIR=$EXECUTORCH_INSTALL/lib/cmake/ExecuTorch \
#     -DEXECUTORCH_ROOT=<path-to-executorch-src>
#   cmake --build . -j$(nproc)

cmake_minimum_required(VERSION 3.19)
project(flux2_runner)
set(CMAKE_CXX_STANDARD 17)

# ExecuTorch source root (for Utils.cmake + include paths)
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../executorch)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(executorch CONFIG REQUIRED)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Sources (monolithic single-file runner)
add_executable(flux2_runner ${CMAKE_CURRENT_LIST_DIR}/flux2_main.cpp)
target_include_directories(flux2_runner PUBLIC ${_common_include_directories})

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
set(link_libraries
  executorch
  executorch_core
  extension_module_static
  extension_tensor
  extension_data_loader
)

if(TARGET extension_flat_tensor)
  list(APPEND link_libraries extension_flat_tensor)
endif()

target_link_options_shared_lib(executorch)

# QNN backend — only link if explicitly requested. Even when the executorch
# install exports qnn_executorch_backend, adding it pulls a hard NEEDED
# dependency on libqnn_executorch_backend.so on device. For the XNNPACK-only
# .pte flow we want the runner to be self-contained.
option(ENABLE_QNN "Link QNN backend (requires QNN HTP runtime on device)" OFF)
if(ENABLE_QNN AND TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
  target_link_options_shared_lib(qnn_executorch_backend)
endif()

# Portable / optimized CPU kernels (fallback for ops not delegated)
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries
    optimized_native_cpu_ops_lib optimized_kernels
    portable_kernels cpublas eigen_blas)
  target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  target_link_options_shared_lib(portable_ops_lib)
endif()

if(TARGET quantized_ops_lib)
  target_link_options_shared_lib(quantized_ops_lib)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
endif()

# XNNPACK backend
set(XNNPACK_ROOT ${EXECUTORCH_ROOT}/backends/xnnpack)

include(CMakeDependentOption)
cmake_dependent_option(
  EXECUTORCH_BUILD_PTHREADPOOL "Build pthreadpool library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF)
cmake_dependent_option(
  EXECUTORCH_BUILD_CPUINFO "Build cpuinfo library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF)

if(EXECUTORCH_BUILD_PTHREADPOOL AND TARGET extension_threadpool)
  list(APPEND link_libraries extension_threadpool pthreadpool)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/pthreadpool/include)
endif()

if(EXECUTORCH_BUILD_CPUINFO AND TARGET cpuinfo)
  list(APPEND link_libraries cpuinfo)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/cpuinfo/include)
endif()

if(TARGET xnnpack_backend)
  set(xnnpack_libs xnnpack_backend XNNPACK microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_libs})
  target_link_options_shared_lib(xnnpack_backend)
endif()

if(ANDROID)
  list(APPEND link_libraries log)
endif()

target_link_libraries(flux2_runner ${link_libraries})
target_compile_options(flux2_runner PUBLIC ${_common_compile_options})
set_target_properties(
  flux2_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
