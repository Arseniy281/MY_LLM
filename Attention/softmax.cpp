#include "softmax.h"
#include <cmath>

Tensor Softmax::forward(const Tensor& matrix) {
    std::vector<size_t> shape = matrix.GetShape();
    result_ = std::make_shared<Tensor>(shape);
    const float* data = matrix.RawData();
    float* result_data = result_->RawData();
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
    return *result_;
}

Tensor Softmax::backward(const Tensor& grad_output) {
    const float* y_data = result_->RawData();
    const float* grad_data = grad_output.RawData();
    
    std::vector<size_t> shape = result_->GetShape();
    size_t embed_dim = shape.back();
    size_t total_size = result_->GetSize();
    size_t num_vectors = total_size / embed_dim;
    
    std::vector<size_t> outer_shape = shape;
    outer_shape.pop_back();
    
    Tensor grad_input(shape);
    float* grad_input_data = grad_input.RawData();
    
    for (size_t vec_idx = 0; vec_idx < num_vectors; vec_idx++) {
        std::vector<size_t> coords = Tensor::IndexToCoord(vec_idx, outer_shape);
        
        float sum = 0.0f;
        for (size_t d = 0; d < embed_dim; d++) {
            std::vector<size_t> full_coords = coords;
            full_coords.push_back(d);
            size_t idx = Tensor::CoordToIndex(full_coords, shape);
            sum += grad_data[idx] * y_data[idx];
        }
        
        for (size_t d = 0; d < embed_dim; d++) {
            std::vector<size_t> full_coords = coords;
            full_coords.push_back(d);
            size_t idx = Tensor::CoordToIndex(full_coords, shape);
            grad_input_data[idx] = y_data[idx] * (grad_data[idx] - sum);
        }
    }
    
    return grad_input;
}