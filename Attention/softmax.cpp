#include <softmax.h>
#include <cmath>

Tensor Softmax::forward(const Tensor& matrix) {
    std::vector<size_t> shape = matrix.GetShape();
    Tensor result(shape);
    const float* data = matrix.RawData();
    float* result_data = result.RawData();
    size_t total_elements = 1;
    for (size_t i = 0; i < shape.size() - 1; i++) {
        total_elements *= shape[i];
    }
    size_t last_dim = shape[shape.size() - 1];
    for (size_t i = 0; i < total_elements; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < last_dim; j++) {
            sum += std::exp(data[i * last_dim + j]);
        }
        for (size_t j = 0; j < last_dim; j++) {
            result_data[i * last_dim + j] = std::exp(data[i * last_dim + j]) / sum;
        }
    }
    return result;
}