cmake_minimum_required(VERSION 3.10)
project(Phase10_Transformer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(transformer_block_test
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Normalization/rms_norm.cpp
    ../Attention/attention.cpp
    ../Layers/feed_forward.cpp
    ../Layers/linear_layer.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sum_op.cpp
    ../Autograd/relu.cpp
    ../Autograd/sigmoid.cpp
    ../Autograd/tanh_op.cpp
    ../Attention/softmax.cpp
    ../Autograd/gelu.cpp
    ../Matmul/matmul.cpp
    ../Attention/rope.cpp
    ../Normalization/rms_norm.cpp
    transformer_block.cpp
    transformer_block_test.cpp
)

add_executable(transformer_test
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Normalization/rms_norm.cpp
    ../Attention/attention.cpp
    ../Layers/feed_forward.cpp
    ../Layers/linear_layer.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sum_op.cpp
    ../Autograd/relu.cpp
    ../Autograd/sigmoid.cpp
    ../Autograd/tanh_op.cpp
    ../Attention/softmax.cpp
    ../Autograd/gelu.cpp
    ../Matmul/matmul.cpp
    ../Attention/rope.cpp
    ../Normalization/rms_norm.cpp
    transformer_block.cpp
    transformer.cpp
    transformer_test.cpp
)

target_include_directories(transformer_block_test PRIVATE .)
target_include_directories(transformer_test PRIVATE .)