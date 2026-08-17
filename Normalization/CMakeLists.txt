cmake_minimum_required(VERSION 3.10)
project(Phase9_Normalization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(normalization_test
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    rms_norm.cpp
    rms_test.cpp
)

target_include_directories(normalization_test PRIVATE .)