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
            float teta = (float)(pos) * std::pow(10000.0f, (-2.0f * i) / head_dim);
            for (size_t b = 0; b < batch; b++) {
                float x = tensor.at({b, pos, i});
                float y = tensor.at({b, pos, i + 1});
                result.at({b, pos, i}) = x * std::cos(teta) - y * std::sin(teta);
                result.at({b, pos, i + 1}) = x * std::sin(teta) + y * std::cos(teta);
            }
        }
    }
    return result;
}