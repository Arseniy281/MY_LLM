cmake_minimum_required(VERSION 3.10)
project(Phase2_MatMul)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul_tests 
    matmul.cpp 
    matmul_tests.cpp
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
)

target_include_directories(matmul_tests PRIVATE .)