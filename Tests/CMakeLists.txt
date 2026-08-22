cmake_minimum_required(VERSION 3.10)
project(All_test)

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
    ../Attention/rope.cpp
    ../Layers/linear_layer.cpp
    ../Attention/softmax.cpp
    ../Attention/attention.cpp
    attention_test.cpp
)

add_executable(autograd_test
    ../Matmul/matmul.cpp 
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Autograd/add_op.cpp
    ../Autograd/mul_op.cpp
    ../Autograd/square_op.cpp
    ../Autograd/sub_op.cpp
    ../Autograd/sum_op.cpp
    autograd_test.cpp
)

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
    ../Layers/mse_loss.cpp
    ../Layers/linear_layer.cpp
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
    ../Layers/mse_loss.cpp
    ../Layers/linear_layer.cpp
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
    ../Layers/mse_loss.cpp
    ../Layers/linear_layer.cpp
    ../Layers/feed_forward.cpp
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
    ../Layers/mse_loss.cpp
    ../Layers/language_model.cpp
    language_model_test.cpp
)

add_executable(sl_test
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
    ../Layers/mse_loss.cpp
    ../Layers/language_model.cpp
    save_load_test.cpp
)

add_executable(matmul_test
    ../Matmul/matmul.cpp 
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    matmul_test.cpp
)

add_executable(normalization_test
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    ../Normalization/rms_norm.cpp
    rms_test.cpp
)

add_executable(tensor_test
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    tensor_test.cpp 
)

add_executable(broadcast_test
    ../Tensor/tensor.cpp
    ../Tensor/tensor_ops.cpp
    ../Tensor/tensor_print.cpp
    broadcast_test.cpp
)

add_executable(tokenizer_test
    ../Tokenizer/bpe_tokenizer.cpp
    ../Tokenizer/trie.cpp
    tokenizer_test.cpp
)

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
    ../Transformer/transformer_block.cpp
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
    ../Transformer/transformer_block.cpp
    ../Transformer/transformer.cpp
    transformer_test.cpp
)

add_executable(final_model_test
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
    ../Transformer/transformer_block.cpp
    ../Tokenizer/bpe_tokenizer.cpp
    ../Tokenizer/trie.cpp
    ../Layers/language_model.cpp
    ../Transformer/transformer.cpp
    ../Autograd/embedding_layer.cpp
    final_model_test.cpp
)

add_executable(tests
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
    ../Transformer/transformer_block.cpp
    ../Tokenizer/bpe_tokenizer.cpp
    ../Tokenizer/trie.cpp
    ../Layers/language_model.cpp
    ../Transformer/transformer.cpp
    ../Autograd/embedding_layer.cpp
    tests.cpp
)

add_executable(text_test
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
    ../Transformer/transformer_block.cpp
    ../Tokenizer/bpe_tokenizer.cpp
    ../Tokenizer/trie.cpp
    ../Layers/language_model.cpp
    ../Transformer/transformer.cpp
    ../Autograd/embedding_layer.cpp
    text_test.cpp
)



target_include_directories(transformer_block_test PRIVATE .)
target_include_directories(transformer_test PRIVATE .)
target_include_directories(tokenizer_test PRIVATE .)
target_include_directories(tensor_test PRIVATE .)
target_include_directories(broadcast_test PRIVATE .)
target_include_directories(normalization_test PRIVATE .)
target_include_directories(matmul_test PRIVATE .)
target_include_directories(attention_test PRIVATE .)
target_include_directories(autograd_test PRIVATE .)
target_include_directories(xor_test PRIVATE .)
target_include_directories(log_test PRIVATE .)
target_include_directories(ffn_test PRIVATE .)
target_include_directories(lm_test PRIVATE .)
target_include_directories(sl_test PRIVATE .)
target_include_directories(final_model_test PRIVATE .)
target_include_directories(tests PRIVATE .)
target_include_directories(text_test PRIVATE .)