#include "mul_op.h"
#include "../Tensor/tensor.h"
#include "../Matmul/matmul.h"
#include <vector>


Tensor MulOp::forward(const std::vector<Tensor*>& inputs) {
    Tensor tensor = MatMul(*inputs[0], *inputs[1]);
    first_ = inputs[0];
    second_ = inputs[1];
    tensor.grad_fn_ = this;
    return tensor;
}

void MulOp::backward(const Tensor& grad_output) const {
    Tensor grad_input(MatMul(grad_output, second_->Transpose()));
    if (first_->grad_ != nullptr) {
        *first_->grad_ += grad_input;
    } else {
        first_->grad_ = new Tensor(grad_input);
    }

    if (first_->grad_fn_ != nullptr) {
        first_->grad_fn_->backward(grad_input);
    }

    grad_input = MatMul(first_->Transpose(), grad_output);

    if (second_->grad_ != nullptr) {
        *second_->grad_ += grad_input;
    } else {
        second_->grad_ = new Tensor(grad_input);
    }

    if (second_->grad_fn_ != nullptr) {
        second_->grad_fn_->backward(grad_input);
    }
}