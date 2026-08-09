#include "tanh_op.h"
#include "../Tensor/tensor.h"
#include <cmath>
#include <memory>

std::shared_ptr<Tensor> TanhOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    parent_ = inputs[0];
    const float* input_data = parent_->RawData();
    size_t size = parent_->GetSize();
    
    std::vector<float> new_data(size);
    for (size_t i = 0; i < size; i++) {
        new_data[i] = std::tanh(input_data[i]);
    }
    
    auto result = std::make_shared<Tensor>(parent_->GetShape(), new_data);
    result->grad_fn_ = this;
    return result;
}

void TanhOp::backward(const Tensor& grad_output) {
    const float* input_data = parent_->RawData();
    const float* grad_data = grad_output.RawData();
    size_t size = parent_->GetSize();
    
    std::vector<float> grad_input_data(size);
    for (size_t i = 0; i < size; i++) {
        float t = std::tanh(input_data[i]);
        grad_input_data[i] = grad_data[i] * (1.0f - t * t);
    }
    
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