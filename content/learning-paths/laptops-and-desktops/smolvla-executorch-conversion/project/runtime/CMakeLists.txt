cmake_minimum_required(VERSION 3.24)
project(smolvla_runner LANGUAGES CXX)

find_package(executorch CONFIG REQUIRED)

add_executable(smolvla_runner main.cpp)
target_compile_features(smolvla_runner PRIVATE cxx_std_17)
target_compile_options(smolvla_runner PRIVATE -O3)
target_link_libraries(
  smolvla_runner
  PRIVATE
    executorch
    extension_module_static
    extension_tensor
    extension_threadpool
    xnnpack_backend
    optimized_native_cpu_ops_lib
)
if(TARGET kleidiai)
  target_link_libraries(smolvla_runner PRIVATE kleidiai)
endif()

