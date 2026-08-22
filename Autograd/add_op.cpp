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
    Tensor grad_first = grad_output;
    Tensor grad_second = grad_output;

    std::vector<size_t> first_shape = first_->GetShape();
    std::vector<size_t> second_shape = second_->GetShape();
    std::vector<size_t> grad_shape = grad_output.GetShape();

    while (grad_first.GetRank() > first_shape.size()) {
        grad_first = grad_first.SumAxis(0);
    }

    for (int axis = static_cast<int>(first_shape.size()) - 1; axis >= 0; axis--) {
        if (first_shape[axis] == 1 && grad_first.GetShape()[axis] > 1) {
            grad_first = grad_first.SumAxis(axis);
        }
    }

    first_->AddGrad(grad_first);
    if (first_->GradFn() != nullptr) {
        first_->GradFn()->backward(grad_first);
    }

    while (grad_second.GetRank() > second_shape.size()) {
        grad_second = grad_second.SumAxis(0);
    }

    for (int axis = static_cast<int>(second_shape.size()) - 1; axis >= 0; axis--) {

        if (second_shape[axis] == 1 && grad_second.GetShape()[axis] > 1) {
            grad_second = grad_second.SumAxis(axis);
        }
    }
    second_->AddGrad(grad_second);
    if (second_->GradFn() != nullptr) {
        second_->GradFn()->backward(grad_second);
    }
    return grad_first;
}