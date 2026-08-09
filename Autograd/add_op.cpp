#include "add_op.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>

Tensor SumAxis(const Tensor& tensor, int axis) {
    if (axis == 0) {
        Tensor result({1, tensor.GetShape()[1]});
        for (size_t j = 0; j < tensor.GetShape()[1]; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < tensor.GetShape()[0]; i++) {
                sum += tensor.at({i, j});
            }
            result.at({0, j}) = sum;
        }
        return result;
    } else if (axis == 1) {
        Tensor result({tensor.GetShape()[0], 1});
        for (size_t i = 0; i < tensor.GetShape()[0]; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < tensor.GetShape()[1]; j++) {
                sum += tensor.at({i, j});
            }
            result.at({i, 0}) = sum;
        }
        return result;
    }
    throw std::runtime_error("SumAxis only supports axis 0 and 1");
}


std::shared_ptr<Tensor> AddOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    first_ = inputs[0];
    second_ = inputs[1];
    first_shape_ = first_->GetShape();
    second_shape_ = second_->GetShape();
    final_shape_ = {std::max(first_shape_[0], second_shape_[0]), std::max(first_shape_[1], second_shape_[1])};
    Tensor tensor = *first_ + *second_;
    tensor.grad_fn_ = this;
    return std::make_shared<Tensor>(tensor);
}

void AddOp::backward(const Tensor& grad_output) {
    Tensor grad_input(grad_output);
    if (first_shape_[0] == 1 && final_shape_[0] > 1) {
        grad_input = SumAxis(grad_input, 0);
    }
    if (first_shape_[1] == 1 && final_shape_[1] > 1) {
        grad_input = SumAxis(grad_input, 1);
    }

    if (first_->grad_ != nullptr) {
        *first_->grad_ += grad_input;
    } else {
        first_->grad_ = std::make_shared<Tensor>(grad_input);
    }

    if (first_->grad_fn_ != nullptr) {
        first_->grad_fn_->backward(grad_input);
    }

    grad_input = grad_output;
    
    if (second_shape_[0] == 1 && final_shape_[0] > 1) {
        grad_input = SumAxis(grad_input, 0);
    }
    if (second_shape_[1] == 1 && final_shape_[1] > 1) {
        grad_input = SumAxis(grad_input, 1);
    }

    if (second_->grad_ != nullptr) {
        *second_->grad_ += grad_input;
    } else {
        second_->grad_ = std::make_shared<Tensor>(grad_input);
    }

    if (second_->grad_fn_ != nullptr) {
        second_->grad_fn_->backward(grad_input);
    }
}