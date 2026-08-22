#include "transformer.h"
#include "../Tensor/tensor.h"
#include "transformer_block.h"

Transformer::Transformer(size_t n, size_t embed_dim, 
    size_t num_heads, size_t hidden_dim) {

    blocks_.reserve(n);
    for (size_t i = 0; i < n; i++) {
        blocks_.emplace_back(embed_dim, num_heads, hidden_dim);
    }
    blocks_count_ = n;
}

std::shared_ptr<Tensor> Transformer::forward(const Tensor& x) {
    auto input = std::make_shared<Tensor>(x);
    for (size_t i = 0; i < blocks_count_; i++) {
        input = blocks_[i].forward(*input);
    }
    return input;
}

Tensor Transformer::backward(const Tensor& grad_output) {
    Tensor grad_x = grad_output;
    for (int i = (int)blocks_count_ - 1; i >= 0; i--) {
        grad_x = blocks_[i].backward(grad_x);
    }
    return grad_x;
}

void Transformer::Update(float lr) {
    for (auto& block : blocks_) {
        block.Update(lr);
    }
}

void Transformer::ClearGrad() {
    for (auto& block : blocks_) {
        block.ClearGrad();
    }
}

void Transformer::ScaleGrad(float factor) {
    for (auto& block : blocks_) {
        block.ScaleGrad(factor);
    }
}

void Transformer::Save(const std::string& folder) const {
    for (size_t i = 0; i < blocks_count_; i++) {
        blocks_[i].Save(folder + "/block_" + std::to_string(i));
    }
}

void Transformer::Load(const std::string& folder) {
    for (size_t i = 0; i < blocks_count_; i++) {
        blocks_[i].Load(folder + "/block_" + std::to_string(i));
    }
}

void Transformer::ResetCache() {
    for (auto& block : blocks_) {
        block.ResetCache();
    }
}

void Transformer::SetUseKVCache(bool value) {
    for (auto& block : blocks_) {
        block.SetUseKVCache(value);
    }
}