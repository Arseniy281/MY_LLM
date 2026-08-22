#include "gelu.h"
#include "../Tensor/tensor.h"
#include "../Matmul/matmul.h"
#include <vector>
#include <memory>
#include <cmath>
#include <numbers>


std::shared_ptr<Tensor> Gelu::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const float pi = 3.141592653589793f;
    parent_ = inputs[0];
    const float* input_data = parent_->RawData();
    std::vector<float> new_data(parent_->GetSize());
    for (size_t i = 0; i < parent_->GetSize(); i++) {
        float x = input_data[i];
        new_data[i] = 0.5 * x * (1 + std::tanh(std::sqrt(2.0f / pi) 
        * (x + 0.044715 * x * x * x)));
    }
    
    auto result = std::make_shared<Tensor>(parent_->GetShape(), new_data);
    result->SetGradFn(this);
    return result;
}

Tensor Gelu::backward(const Tensor& grad_output) {
    const float* grad_data = grad_output.RawData();
    size_t size = parent_->GetSize();

    const float pi = 3.141592653589793f;
    const float sqrt_2_pi = std::sqrt(2.0f / pi);
    const float a = 0.044715f;
    
    std::vector<float> grad_input_data(size);
    for (size_t i = 0; i < size; i++) {
        float x = parent_->at(i);
        float x3 = x * x * x;
        float arg = sqrt_2_pi * (x + a * x3);
        float tanh_val = std::tanh(arg);
        float sech = 1.0f / std::cosh(arg);
        float sech_sq = sech * sech;
        
        float gelu_derivative = 0.5f * (1.0f + tanh_val) 
             + 0.5f * x * sech_sq * sqrt_2_pi * (1.0f + 3.0f * a * x * x);

        grad_input_data[i] = grad_data[i] * gelu_derivative;
    }
    Tensor grad_input(parent_->GetShape(), grad_input_data);

    parent_->AddGrad(grad_input);


    if (parent_->GradFn() != nullptr) {
        parent_->GradFn()->backward(grad_input);
    }

    return grad_input;
}