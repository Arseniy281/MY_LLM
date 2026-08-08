#include "square_op.h"
#include "../Tensor/tensor.h"
#include <vector>

Tensor SquareOp::forward(const std::vector<Tensor*>& inputs) {
    Tensor tensor = *inputs[0] * (*inputs[0]);
    parent_ = inputs[0];
    tensor.grad_fn_ = this;
    return tensor;
}

void SquareOp::backward(const Tensor& grad_output) const {
    Tensor grad_input(grad_output * (*parent_) * 2);

    if (parent_->grad_ != nullptr) {
        *parent_->grad_ += grad_input;
    } else {
        parent_->grad_ = new Tensor(grad_input);
    }

    if (parent_->grad_fn_ != nullptr) {
        parent_->grad_fn_->backward(grad_input);
    }
}