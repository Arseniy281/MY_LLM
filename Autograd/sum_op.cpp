#include "sum_op.h"
#include "../Matmul/matmul.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>

std::shared_ptr<Tensor> SumOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    parent_ = inputs[0];
    
    float total = 0.0f;
    const float* data = parent_->RawData();
    for (size_t i = 0; i < parent_->GetSize(); i++) {
        total += data[i];
    }
    
    auto result = std::make_shared<Tensor>(Tensor({1, 1}, total));
    result->SetGradFn(this);
    return result;
}

Tensor SumOp::backward(const Tensor& grad_output) {
    Tensor grad_input(parent_->GetShape(), grad_output.RawData()[0]);
    
    if (parent_->Grad() != nullptr) {
        *parent_->Grad() += grad_input;
    } else {
        parent_->Grad() = std::make_shared<Tensor>(grad_input);
    }
    
    if (parent_->GradFn() != nullptr) {
        parent_->GradFn()->backward(grad_input);
    }

    return grad_input;
}