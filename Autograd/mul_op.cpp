#include "mul_op.h"
#include "../Tensor/tensor.h"
#include "../Matmul/matmul.h"
#include <vector>
#include <memory>


std::shared_ptr<Tensor> MulOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    first_ = inputs[0];
    second_ = inputs[1];
    
    auto result = std::make_shared<Tensor>(MatMul(*first_, *second_));
    result->grad_fn_ = this;
    
    
    return result;
}

void MulOp::backward(const Tensor& grad_output) {
    Tensor grad_first = MatMul(grad_output, second_->Transpose());
    
    if (first_->grad_ != nullptr) {
        *first_->grad_ += grad_first;
    } else {
        first_->grad_ = std::make_shared<Tensor>(grad_first);
    }
    
    if (first_->grad_fn_ != nullptr) {
        first_->grad_fn_->backward(grad_first);
    }
    
    Tensor grad_second = MatMul(first_->Transpose(), grad_output);
    
    if (second_->grad_ != nullptr) {
        *second_->grad_ += grad_second;
    } else {
        second_->grad_ = std::make_shared<Tensor>(grad_second);
    }
    
    if (second_->grad_fn_ != nullptr) {
        second_->grad_fn_->backward(grad_second);
    }
}