#include "sub_op.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>

std::shared_ptr<Tensor> SubOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    first_ = inputs[0];
    second_ = inputs[1];
    
    auto result = std::make_shared<Tensor>(*first_ - *second_);
    result->SetGradFn(this);
    return result;
}

Tensor SubOp::backward(const Tensor& grad_output) {
    first_->AddGrad(grad_output);

    if (first_->GradFn() != nullptr) {
        first_->GradFn()->backward(grad_output);
    }

    second_->AddGrad(-grad_output);

    if (second_->GradFn() != nullptr) {
        second_->GradFn()->backward(-grad_output);
    }

    return grad_output;
}