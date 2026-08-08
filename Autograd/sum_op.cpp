#include "sum_op.h"
#include "../Matmul/matmul.h"
#include "../Tensor/tensor.h"
#include <vector>

Tensor SumOp::forward(const std::vector<Tensor*>& inputs) {
    float* data = inputs[0]->data();
    float total = 0.0f;
    for (size_t i = 0; i < inputs[0]->GetSize(); i++) {
        total += data[i];
    }
    Tensor tensor({1, 1}, total);
    parent_ = inputs[0];
    tensor.grad_fn_ = this;
    return tensor;
}

void SumOp::backward(const Tensor& grad_output) const {
    Tensor grad_input(parent_->GetShape(), grad_output.data()[0]);
    
    if (parent_->grad_ != nullptr) {
        *parent_->grad_ += grad_input;
    } else {
        parent_->grad_ = new Tensor(grad_input);
    }
    
    if (parent_->grad_fn_ != nullptr) {
        parent_->grad_fn_->backward(grad_input);
    }
}