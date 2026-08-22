#include "square_op.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>

std::shared_ptr<Tensor> SquareOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    parent_ = inputs[0];
    
    auto result = std::make_shared<Tensor>(*parent_ * *parent_);
    result->SetGradFn(this);
    return result;
}

Tensor SquareOp::backward(const Tensor& grad_output) {
    Tensor grad_input(grad_output * (*parent_) * 2);

    parent_->AddGrad(grad_input);

    if (parent_->GradFn() != nullptr) {
        parent_->GradFn()->backward(grad_input);
    }

    return grad_input;
}