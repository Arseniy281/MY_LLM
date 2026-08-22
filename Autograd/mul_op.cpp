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
    first_->AddGrad(grad_first);

    if (first_->GradFn() != nullptr) {
        first_->GradFn()->backward(grad_first);
    }
    std::vector<size_t> first_shape = first_->GetShape();
    Tensor grad_second(second_->GetShape(), 0.0f);


    if (first_shape.size() == 3 && second_->GetShape().size() == 2) {
        size_t batch = first_shape[0];
        size_t seq_len = first_shape[1];
        size_t in = first_shape[2];
        size_t out = second_->GetShape()[1];

        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t i = 0; i < in; i++) {
                    float x = first_->at({b, s, i});
                    for (size_t j = 0; j < out; j++) {
                        grad_second.at({i, j}) += x * grad_output.at({b, s, j});
                    }
                }
            }
        }
    }
    else {
        grad_second = MatMul(first_->Transpose(), grad_output);
    }

    second_->AddGrad(grad_second);
    if (second_->GradFn() != nullptr) {
        second_->GradFn()->backward(grad_second);
    }

    return grad_first;
}