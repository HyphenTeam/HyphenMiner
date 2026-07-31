cmake_minimum_required(VERSION 3.25)
project(hyphen_accelerators LANGUAGES C CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Native plugin build type" FORCE)
endif()

option(HYPHEN_ENABLE_CUDA "Build the NVIDIA CUDA backend" OFF)
option(HYPHEN_ENABLE_HIP "Build the AMD HIP backend" OFF)
option(HYPHEN_ENABLE_OPENVINO "Build the Intel OpenVINO NPU/GPU backend" OFF)
option(HYPHEN_ENABLE_QNN "Build the Qualcomm QNN backend" OFF)

set(HYPHEN_PLUGIN_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../accelerators"
    CACHE PATH "Directory for Hyphen accelerator plugins")

function(hyphen_configure_plugin target output_name)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set_target_properties(${target} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${output_name}"
    RUNTIME_OUTPUT_DIRECTORY "${HYPHEN_PLUGIN_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${HYPHEN_PLUGIN_OUTPUT_DIR}")
endfunction()

if(HYPHEN_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(hyphen_cuda SHARED cuda/hyphen_cuda.cu)
  target_compile_features(hyphen_cuda PRIVATE cxx_std_17 cuda_std_17)
  set_target_properties(hyphen_cuda PROPERTIES CUDA_ARCHITECTURES native)
  hyphen_configure_plugin(hyphen_cuda hyphen_backend_cuda)
endif()

if(HYPHEN_ENABLE_HIP)
  find_package(hip CONFIG REQUIRED)
  add_library(hyphen_hip SHARED hip/hyphen_hip.cpp)
  target_compile_features(hyphen_hip PRIVATE cxx_std_17)
  target_link_libraries(hyphen_hip PRIVATE hip::host)
  set_source_files_properties(hip/hyphen_hip.cpp PROPERTIES LANGUAGE HIP)
  hyphen_configure_plugin(hyphen_hip hyphen_backend_hip)
endif()

if(HYPHEN_ENABLE_OPENVINO)
  find_package(OpenVINO CONFIG REQUIRED COMPONENTS Runtime)
  add_library(hyphen_openvino SHARED openvino/hyphen_openvino.cpp)
  target_compile_features(hyphen_openvino PRIVATE cxx_std_17)
  target_link_libraries(hyphen_openvino PRIVATE openvino::runtime)
  hyphen_configure_plugin(hyphen_openvino hyphen_backend_openvino)
endif()

if(HYPHEN_ENABLE_QNN)
  if(NOT DEFINED QNN_SDK_ROOT)
    message(FATAL_ERROR "Set QNN_SDK_ROOT to the Qualcomm AI Engine Direct SDK root")
  endif()
  add_library(hyphen_qnn SHARED qnn/hyphen_qnn.cpp)
  target_compile_features(hyphen_qnn PRIVATE cxx_std_17)
  target_include_directories(hyphen_qnn PRIVATE "${QNN_SDK_ROOT}/include/QNN")
  hyphen_configure_plugin(hyphen_qnn hyphen_backend_qnn)
endif()
