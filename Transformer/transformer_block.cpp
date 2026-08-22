#include "transformer_block.h"
#include "../Tensor/tensor.h"
#include <sys/stat.h>
#include <errno.h> 
#include <vector>

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim)
        : rms_norm_1_(embed_dim),
          attention_(embed_dim, num_heads),
          rms_norm_2_(embed_dim),
          feed_forward_(embed_dim, hidden_dim) {}

std::shared_ptr<Tensor> TransformerBlock::forward(const Tensor& x) {
    auto input = std::make_shared<Tensor>(x);
    Tensor residual = x;
    
    *input = rms_norm_1_.forward(*input);
    saved_norm_1_input_ = *input;
    input = attention_.forward(*input);
    saved_attention_input_ = *input;
    *input += residual;
    residual = *input;
    saved_residual_1_ = residual;
    
    *input = rms_norm_2_.forward(*input);
    saved_norm_2_input_ = *input;
    *input = feed_forward_.forward(*input);
    saved_ffn_input_ = *input;
    *input += residual;
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
    grad_x += grad_residual_1;

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

void TransformerBlock::ScaleGrad(float factor) {
    rms_norm_1_.ScaleGrad(factor);
    attention_.ScaleGrad(factor);
    rms_norm_2_.ScaleGrad(factor);
    feed_forward_.ScaleGrad(factor);
}

void TransformerBlock::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + folder);
    }

    rms_norm_1_.Save(folder + "/rmsnorm_1_gamma");
    attention_.Save(folder + "/attention");
    rms_norm_2_.Save(folder + "/rmsnorm_2_gamma");
    feed_forward_.Save(folder + "/feedforward");
}

void TransformerBlock::Load(const std::string& folder) {
    rms_norm_1_.Load(folder + "/rmsnorm_1_gamma");
    attention_.Load(folder + "/attention");
    rms_norm_2_.Load(folder + "/rmsnorm_2_gamma");
    feed_forward_.Load(folder + "/feedforward");
}

void TransformerBlock::ResetCache() {
    attention_.ResetCache();
}

void TransformerBlock::SetUseKVCache(bool value) {
    attention_.SetUseKVCache(value);
}