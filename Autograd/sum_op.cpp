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
    result->grad_fn_ = this;
    return result;
}

void SumOp::backward(const Tensor& grad_output) {
    Tensor grad_input(parent_->GetShape(), grad_output.RawData()[0]);
    
    if (parent_->grad_ != nullptr) {
        *parent_->grad_ += grad_input;
    } else {
        parent_->grad_ = std::make_shared<Tensor>(grad_input);
    }
    
    if (parent_->grad_fn_ != nullptr) {
        parent_->grad_fn_->backward(grad_input);
    }
}