#include "sigmoid.h"
#include "../Tensor/tensor.h"
#include "../Matmul/matmul.h"
#include <vector>
#include <cmath>
#include <memory>


std::shared_ptr<Tensor> Sigmoid::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    parent_ = inputs[0];
    const float* input_data = parent_->RawData();
    std::vector<float> new_data(parent_->GetSize());
    for (size_t i = 0; i < parent_->GetSize(); i++) {
        new_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
    }
    
    auto result = std::make_shared<Tensor>(parent_->GetShape(), new_data);
    result->grad_fn_ = this;
    return result;
}

void Sigmoid::backward(const Tensor& grad_output) {
    const float* input_data = parent_->RawData();
    const float* grad_data = grad_output.RawData();
    size_t size = parent_->GetSize();
    
    std::vector<float> grad_input_data(size);
    float avg_s = 0.0f;
    float avg_grad = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float s = 1.0f / (1.0f + std::exp(-input_data[i]));
        avg_s += s;
        avg_grad += grad_data[i];
        grad_input_data[i] = grad_data[i] * s * (1.0f - s);
    }
    avg_s /= size;
    avg_grad /= size;
    
    Tensor grad_input(parent_->GetShape(), grad_input_data);

    if (parent_->grad_ != nullptr) {
        *parent_->grad_ += grad_input;
    } else {
        parent_->grad_ = std::make_shared<Tensor>(grad_input);
    }

    if (parent_->grad_fn_ != nullptr) {
        parent_->grad_fn_->backward(grad_input);
    }
}