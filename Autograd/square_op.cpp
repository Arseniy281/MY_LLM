#include "square_op.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>

std::shared_ptr<Tensor> SquareOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    parent_ = inputs[0];
    
    auto result = std::make_shared<Tensor>(*parent_ * *parent_);
    result->grad_fn_ = this;
    return result;
}

void SquareOp::backward(const Tensor& grad_output) {
    Tensor grad_input(grad_output * (*parent_) * 2);

    if (parent_->grad_ != nullptr) {
        *parent_->grad_ += grad_input;
    } else {
        parent_->grad_ = std::make_shared<Tensor>(grad_input);
    }

    if (parent_->grad_fn_ != nullptr) {
        parent_->grad_fn_->backward(grad_input);
    }
}