#include "rope.h"
#include "../Tensor/tensor.h"
#include <iostream>
#include <cmath>

Tensor RoPE(const Tensor& tensor) {
    size_t batch = tensor.GetShape()[0];
    size_t seq_len = tensor.GetShape()[1];
    size_t head_dim = tensor.GetShape()[2];
    Tensor result = tensor;
    for (size_t pos = 0; pos < seq_len; pos++) {
        for (size_t i = 0; i < head_dim; i += 2) {
            float theta = (float)(pos) * std::pow(10000.0f, (-2.0f * i) / head_dim);
            for (size_t b = 0; b < batch; b++) {
                float x = tensor.at({b, pos, i});
                float y = tensor.at({b, pos, i + 1});
                result.at({b, pos, i}) = x * std::cos(theta) - y * std::sin(theta);
                result.at({b, pos, i + 1}) = x * std::sin(theta) + y * std::cos(theta);
            }
        }
    }
    return result;
}

Tensor RoPE_backward(const Tensor& tensor) {
    size_t batch = tensor.GetShape()[0];
    size_t seq_len = tensor.GetShape()[1];
    size_t head_dim = tensor.GetShape()[2];
    Tensor result = tensor;
    
    for (size_t pos = 0; pos < seq_len; pos++) {
        for (size_t i = 0; i < head_dim; i += 2) {
            float theta = (float)(pos) * std::pow(10000.0f, (-2.0f * i) / head_dim);
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);
            
            for (size_t b = 0; b < batch; b++) {
                float x = tensor.at({b, pos, i});
                float y = tensor.at({b, pos, i + 1});
                
                result.at({b, pos, i}) = x * cos_theta + y * sin_theta;
                result.at({b, pos, i + 1}) = -x * sin_theta + y * cos_theta;
            }
        }
    }
    return result;
}