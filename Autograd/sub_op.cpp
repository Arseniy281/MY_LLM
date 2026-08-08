#include "sub_op.h"
#include "../Tensor/tensor.h"
#include <vector>

Tensor SubOp::forward(const std::vector<Tensor*>& inputs) {
    Tensor tensor = *inputs[0];
    tensor -= *inputs[1];
    first_ = inputs[0];
    second_ = inputs[1];
    tensor.grad_fn_ = this;
    return tensor;
}

void SubOp::backward(const Tensor& grad_output) const {
    if (first_->grad_ != nullptr) {
        *first_->grad_ += grad_output;
    } else {
        first_->grad_ = new Tensor(grad_output);
    }

    if (first_->grad_fn_ != nullptr) {
        first_->grad_fn_->backward(grad_output);
    }

    if (second_->grad_ != nullptr) {
        *second_->grad_ -= grad_output;
    } else {
        second_->grad_ = new Tensor(-grad_output);
    }

    if (second_->grad_fn_ != nullptr) {
        second_->grad_fn_->backward(-grad_output);
    }
}