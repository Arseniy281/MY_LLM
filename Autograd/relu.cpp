#include "relu.h"
#include "../Tensor/tensor.h"
#include "../Matmul/matmul.h"
#include <vector>
#include <memory>


std::shared_ptr<Tensor> ReLU::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    parent_ = inputs[0];
    const float* input_data = parent_->RawData();
    std::vector<float> new_data(parent_->GetSize());
    for (size_t i = 0; i < parent_->GetSize(); i++) {
        new_data[i] = std::max(0.0f, input_data[i]);
    }
    
    auto result = std::make_shared<Tensor>(parent_->GetShape(), new_data);
    result->SetGradFn(this);
    return result;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    const float* input_data = parent_->RawData();
    const float* grad_data = grad_output.RawData();
    size_t size = parent_->GetSize();
    
    std::vector<float> grad_input_data(size);
    for (size_t i = 0; i < size; i++) {
        grad_input_data[i] = (input_data[i] > 0.0f) ? grad_data[i] : 0.0f;
    }
    Tensor grad_input(parent_->GetShape(), grad_input_data);
    grad_input *= grad_output;

    parent_->AddGrad(grad_input);

    if (parent_->GradFn() != nullptr) {
        parent_->GradFn()->backward(grad_input);
    }

    return grad_input;
}