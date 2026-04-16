# FLUX.2-klein-4B ExecuTorch QNN Runner
#
# Build for Android ARM64:
#   mkdir build-android && cd build-android
#   cmake .. \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_PLATFORM=android-30 \
#     -DEXECUTORCH_ROOT=<path-to-executorch> \
#     -DQNN_SDK_ROOT=<path-to-qnn-sdk> \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build . -j$(nproc)

cmake_minimum_required(VERSION 3.19)
project(flux2_runner)
set(CMAKE_CXX_STANDARD 17)

# ExecuTorch root
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

# Find prebuilt ExecuTorch
find_package(executorch CONFIG REQUIRED)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Sources
set(_flux2_runner_srcs
    ${CMAKE_CURRENT_LIST_DIR}/flux2_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/flux2_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/flux2_runner.h
)

# Build executable
add_executable(flux2_runner ${_flux2_runner_srcs})
target_include_directories(flux2_runner PUBLIC ${_common_include_directories})
target_link_libraries(
  flux2_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_module
  extension_tensor
)
target_compile_options(flux2_runner PUBLIC ${_common_compile_options})
set_target_properties(
  flux2_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
