#include "transformer_block.h"
#include "../Tensor/tensor.h"

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim)
        : rms_norm_1_(embed_dim),
          attention_(embed_dim, num_heads),
          rms_norm_2_(embed_dim),
          feed_forward_(embed_dim, hidden_dim) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    Tensor input = x;
    Tensor residual = x;
    
    input = rms_norm_1_.forward(input);
    saved_norm_1_input_ = input;
    input = attention_.forward(input);
    saved_attention_input_ = input;
    input += residual;
    residual = input;
    saved_residual_1_ = residual;
    
    input = rms_norm_2_.forward(input);
    saved_norm_2_input_ = input;
    input = feed_forward_.forward(input);
    saved_ffn_input_ = input;
    input += residual;
    saved_residual_2_ = residual;
    
    return input;
}

Tensor TransformerBlock::backward(const Tensor& grad_output) {
    Tensor grad_residual_2 = grad_output;
    Tensor grad_ffn = feed_forward_.backward(grad_output);
    Tensor grad_norm_2 = rms_norm_2_.backward(grad_ffn);

    Tensor grad_residual_1 = grad_norm_2 + grad_residual_2;
    Tensor grad_attn = attention_.backward(grad_residual_1);
    Tensor grad_x = rms_norm_1_.backward(grad_attn);

    return grad_x;
}

void TransformerBlock::Update(float lr) {
    rms_norm_1_.Update(lr);
    attention_.Update(lr);
    rms_norm_2_.Update(lr);
    feed_forward_.Update(lr);
}

void TransformerBlock::ClearGrad() {
    rms_norm_1_.ClearGrad();
    attention_.ClearGrad();
    rms_norm_2_.ClearGrad();
    feed_forward_.ClearGrad();
}