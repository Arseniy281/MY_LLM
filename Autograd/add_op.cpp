#include "add_op.h"
#include "../Tensor/tensor.h"
#include <vector>


Tensor AddOp::forward(const std::vector<Tensor*>& inputs) {
    first_ = inputs[0];
    second_ = inputs[1];
    Tensor tensor = *first_;
    tensor += *second_;
    tensor.grad_fn_ = this;
    return tensor;
}

void AddOp::backward(const Tensor& grad_output) const {
    for (const auto& parent : {first_, second_}) {
        if (parent->grad_ != nullptr) {
            *(parent->grad_) += grad_output;
        } else {
            parent->grad_ = new Tensor(grad_output);
        }
        
        if (parent->grad_fn_ != nullptr) {
            parent->grad_fn_->backward(grad_output);
        }
    }
}