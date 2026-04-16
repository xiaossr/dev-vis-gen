cmake_minimum_required(VERSION 3.24)
project(flux2_qnn_runner)

option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build the optimized kernels" OFF)

include(CMakeDependentOption)
cmake_dependent_option(
  EXECUTORCH_BUILD_PTHREADPOOL "Build pthreadpool library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF
)
cmake_dependent_option(
  EXECUTORCH_BUILD_CPUINFO "Build cpuinfo library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF
)

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

# --- link libraries ---
set(link_libraries executorch gflags extension_module_static extension_tensor
    extension_data_loader extension_flat_tensor)

# QNN backend
if(TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
  executorch_target_link_options_shared_lib(qnn_executorch_backend)
else()
  message(WARNING "qnn_executorch_backend not found — QNN delegate will not work at runtime")
endif()

# Portable ops (CPU fallback for ops not delegated to HTP)
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib optimized_kernels
       portable_kernels cpublas eigen_blas)
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()

if(TARGET quantized_ops_lib)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
endif()

# XNNPACK for any remaining CPU-delegated ops (optional)
set(XNNPACK_ROOT ${EXECUTORCH_ROOT}/backends/xnnpack)

if(EXECUTORCH_BUILD_PTHREADPOOL)
  list(APPEND link_libraries extension_threadpool pthreadpool)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/pthreadpool/include)
endif()

if(EXECUTORCH_BUILD_CPUINFO)
  list(APPEND link_libraries cpuinfo)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/cpuinfo/include)
endif()

if(TARGET xnnpack_backend)
  set(xnnpack_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_libs})
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()

if(ANDROID)
  list(APPEND link_libraries log)
endif()

# --- executable ---
add_executable(flux2_qnn_main flux2_qnn_main.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  target_link_options_gc_sections(flux2_qnn_main)
  if(NOT APPLE)
    target_link_options(flux2_qnn_main PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(flux2_qnn_main PUBLIC ${_common_include_directories})
target_link_libraries(flux2_qnn_main PUBLIC ${link_libraries})
target_compile_options(flux2_qnn_main PUBLIC ${_common_compile_options})

if(ANDROID)
  set_target_properties(flux2_qnn_main PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'")
elseif(UNIX)
  set_target_properties(flux2_qnn_main PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'")
endif()
