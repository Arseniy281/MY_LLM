cmake_minimum_required(VERSION 3.10)
project(Phase3_AutoGrad)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(autograd_tests 
    ../Matmul/matmul.cpp 
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    autograd_tests.cpp
    add_op.cpp
    mul_op.cpp
    square_op.cpp
    sub_op.cpp
    sum_op.cpp
)

target_include_directories(autograd_tests PRIVATE .)