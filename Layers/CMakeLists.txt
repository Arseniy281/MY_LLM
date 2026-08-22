cmake_minimum_required(VERSION 3.10)
project(Phase4_Layers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(xor_test
    ../Matmul/matmul.cpp 
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/sum_op.cpp
    ../Autograd/relu.cpp
    ../Autograd/sigmoid.cpp
    ../Autograd/tanh_op.cpp
    mse_loss.cpp
    linear_layer.cpp
    xor_test.cpp
)

add_executable(log_test
    ../Matmul/matmul.cpp 
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/sum_op.cpp
    ../Autograd/relu.cpp
    ../Autograd/sigmoid.cpp
    ../Autograd/tanh_op.cpp
    mse_loss.cpp
    linear_layer.cpp
    log_test.cpp
)

add_executable(ffn_test
    ../Matmul/matmul.cpp 
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/sum_op.cpp
    ../Autograd/relu.cpp
    ../Autograd/sigmoid.cpp
    ../Autograd/tanh_op.cpp
    ../Autograd/gelu.cpp
    mse_loss.cpp
    linear_layer.cpp
    feed_forward.cpp
    ffn_test.cpp
)

add_executable(lm_test
    ../Matmul/matmul.cpp 
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/sum_op.cpp
    ../Autograd/relu.cpp
    ../Autograd/sigmoid.cpp
    ../Autograd/tanh_op.cpp
    ../Autograd/gelu.cpp
    ../Attention/softmax.cpp
    ../Autograd/embedding_layer.cpp
    ../Layers/linear_layer.cpp
    ../Transformer/transformer.cpp
    ../Transformer/transformer_block.cpp
    ../Layers/feed_forward.cpp
    ../Normalization/rms_norm.cpp
    ../Attention/attention.cpp
    ../Attention/rope.cpp
    mse_loss.cpp
    language_model.cpp
    language_model_test.cpp
)

add_executable(sv_test
    ../Matmul/matmul.cpp 
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/sum_op.cpp
    ../Autograd/relu.cpp
    ../Autograd/sigmoid.cpp
    ../Autograd/tanh_op.cpp
    ../Autograd/gelu.cpp
    ../Attention/softmax.cpp
    ../Autograd/embedding_layer.cpp
    ../Layers/linear_layer.cpp
    ../Transformer/transformer.cpp
    ../Transformer/transformer_block.cpp
    ../Layers/feed_forward.cpp
    ../Normalization/rms_norm.cpp
    ../Attention/attention.cpp
    ../Attention/rope.cpp
    mse_loss.cpp
    language_model.cpp
    save_load_test.cpp
)

target_include_directories(xor_test PRIVATE .)
target_include_directories(log_test PRIVATE .)
target_include_directories(ffn_test PRIVATE .)
target_include_directories(lm_test PRIVATE .)
target_include_directories(sv_test PRIVATE .)