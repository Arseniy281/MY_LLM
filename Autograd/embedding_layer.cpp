#include "embedding_layer.h"
#include <memory>
#include <vector>

EmbeddingLayer::EmbeddingLayer(size_t vocab_size, size_t embedding_dim) : vocab_size_(vocab_size), embedding_dim_(embedding_dim) {
    embeddings_ = Tensor::Random({vocab_size_, embedding_dim_});
    grad_ = std::make_shared<Tensor>(Tensor({vocab_size_, embedding_dim_}, 0.0f));
}

std::shared_ptr<Tensor> EmbeddingLayer::forward(const std::vector<std::shared_ptr<Tensor>>& indices) {
    indices_ = indices[0];
    size_t indices_count = indices_->GetSize();
    std::vector<float> data;
    float* indices_data = embeddings_.RawData();
    for (size_t i = 0; i < indices_count; i++) {
        size_t idx = indices_->at(i);
        data.insert(data.end(), indices_data + idx * embedding_dim_, indices_data + (idx + 1) * embedding_dim_);
    }
    auto output = std::make_shared<Tensor>(Tensor({indices_count, embedding_dim_}, data));
    output->grad_fn_ = this;
    return output;
}

void EmbeddingLayer::backward(const Tensor& grad_output) {
    size_t batch_size = indices_->GetSize();
    for (size_t i = 0; i < batch_size; i++) {
        size_t idx = indices_->at(i);
        for (size_t j = 0; j < embedding_dim_; j++) {
            grad_->at({idx, j}) += grad_output.at({i, j});
        }
    }

    if (indices_->grad_ != nullptr) {
        *indices_->grad_ += grad_output;
    } else {
        indices_->grad_ = std::make_shared<Tensor>(grad_output);
    }
}

void EmbeddingLayer::ClearGrad() {
    for (size_t i = 0; i < vocab_size_; i++) {
        for (size_t j = 0; j < embedding_dim_; j++) {
            grad_->at({i, j}) = 0.0f;
        }
    }
}

void EmbeddingLayer::Update(float lr) {
    for (size_t i = 0; i < vocab_size_; i++) {
        for (size_t j = 0; j < embedding_dim_; j++) {
            embeddings_.at({i, j}) -= lr * grad_->at({i, j});
        }
    }
}