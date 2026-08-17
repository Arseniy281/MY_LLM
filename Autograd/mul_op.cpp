#include "mul_op.h"
#include "../Tensor/tensor.h"
#include "../Matmul/matmul.h"
#include <vector>
#include <memory>


std::shared_ptr<Tensor> MulOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    first_ = inputs[0];
    second_ = inputs[1];
    
    auto result = std::make_shared<Tensor>(MatMul(*first_, *second_));
    result->SetGradFn(this);
    
    
    return result;
}

Tensor MulOp::backward(const Tensor& grad_output) {
    Tensor grad_first = MatMul(grad_output, second_->Transpose());
    
    if (first_->Grad() != nullptr) {
        *first_->Grad() += grad_first;
    } else {
        first_->Grad() = std::make_shared<Tensor>(grad_first);
    }
    
    if (first_->GradFn() != nullptr) {
        first_->GradFn()->backward(grad_first);
    }
    
    Tensor grad_second = MatMul(first_->Transpose(), grad_output);
    
    if (second_->Grad() != nullptr) {
        *second_->Grad() += grad_second;
    } else {
        second_->Grad() = std::make_shared<Tensor>(grad_second);
    }
    
    if (second_->GradFn() != nullptr) {
        second_->GradFn()->backward(grad_second);
    }

    return grad_first;
}