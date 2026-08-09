#include "sub_op.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>

std::shared_ptr<Tensor> SubOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    first_ = inputs[0];
    second_ = inputs[1];
    
    auto result = std::make_shared<Tensor>(*first_ - *second_);
    result->grad_fn_ = this;
    return result;
}

void SubOp::backward(const Tensor& grad_output) {
    if (first_->grad_ != nullptr) {
        *first_->grad_ += grad_output;
    } else {
        first_->grad_ = std::make_shared<Tensor>(grad_output);
    }

    if (first_->grad_fn_ != nullptr) {
        first_->grad_fn_->backward(grad_output);
    }

    if (second_->grad_ != nullptr) {
        *second_->grad_ -= grad_output;
    } else {
        second_->grad_ = std::make_shared<Tensor>(-grad_output);
    }

    if (second_->grad_fn_ != nullptr) {
        second_->grad_fn_->backward(-grad_output);
    }
}