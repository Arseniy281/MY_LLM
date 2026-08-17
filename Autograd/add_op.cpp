#include "add_op.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>


std::shared_ptr<Tensor> AddOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    first_ = inputs[0];
    second_ = inputs[1];
    first_shape_ = first_->GetShape();
    second_shape_ = second_->GetShape();
    final_shape_ = {std::max(first_shape_[0], second_shape_[0]), std::max(first_shape_[1], second_shape_[1])};
    Tensor tensor = *first_ + *second_;
    tensor.SetGradFn(this);
    return std::make_shared<Tensor>(tensor);
}

Tensor AddOp::backward(const Tensor& grad_output) {
    Tensor grad_first(grad_output);
    if (first_shape_[0] == 1 && final_shape_[0] > 1) {
        grad_first = grad_first.SumAxis(0);
    }
    if (first_shape_[1] == 1 && final_shape_[1] > 1) {
        grad_first = grad_first.SumAxis(1);
    }

    if (first_->Grad() != nullptr) {
        *first_->Grad() += grad_first;
    } else {
        first_->Grad() = std::make_shared<Tensor>(grad_first);
    }

    if (first_->GradFn() != nullptr) {
        first_->GradFn()->backward(grad_first);
    }

    Tensor grad_second(grad_output);
    
    if (second_shape_[0] == 1 && final_shape_[0] > 1) {
        grad_second = grad_second.SumAxis(0);
    }
    if (second_shape_[1] == 1 && final_shape_[1] > 1) {
        grad_second = grad_second.SumAxis(1);
    }

    if (second_->Grad() != nullptr) {
        *second_->Grad() += grad_second;
    } else {
        second_->Grad() = std::make_shared<Tensor>(grad_second);
    }

    if (second_->GradFn() != nullptr) {
        second_->GradFn()->backward(grad_second);
    }

    return grad_first + grad_second;
}