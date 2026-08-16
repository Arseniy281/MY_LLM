cmake_minimum_required(VERSION 3.10)
project(Phase8_Attention)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(attention_test
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Matmul/matmul.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sum_op.cpp
    ../Autograd/relu.cpp
    ../Autograd/sigmoid.cpp
    ../Layers/linear_layer.cpp
    attention.cpp 
    softmax.cpp
    attention_test.cpp
)

target_include_directories(attention_test PRIVATE .)