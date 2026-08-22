#include "softmax.h"
#include <cmath>

Tensor Softmax::forward(const Tensor& matrix) {
    std::vector<size_t> shape = matrix.GetShape();
    result_ = std::make_shared<Tensor>(shape);

    const float* data = matrix.RawData();
    float* result_data = result_->RawData();

    size_t total_elements = 1;
    for (size_t i = 0; i + 1 < shape.size(); i++) {
        total_elements *= shape[i];
    }

    size_t last_dim = shape.back();

    for (size_t i = 0; i < total_elements; i++) {
        float max_value = -std::numeric_limits<float>::infinity();

        for (size_t j = 0; j < last_dim; j++) {
            max_value = std::max(
                max_value,
                data[i * last_dim + j]
            );
        }

        float sum = 0.0f;

        for (size_t j = 0; j < last_dim; j++) {
            float value = std::exp(
                data[i * last_dim + j] - max_value
            );

            result_data[i * last_dim + j] = value;
            sum += value;
        }

        for (size_t j = 0; j < last_dim; j++) {
            result_data[i * last_dim + j] /= sum;
        }
    }

    if (shape.size() == 3) {
        for (size_t b = 0; b < shape[0]; b++) {
            for (size_t row = 0; row < shape[1]; row++) {
                float sum = 0.0f;

                for (size_t j = 0; j < shape[2]; j++) {
                    sum += result_->at({b, row, j});
                }

                if (std::abs(sum - 1.0f) > 1e-4f) {
                    std::cout
                        << "[SOFTMAX ERROR] row "
                        << row
                        << " sum = "
                        << sum
                        << "\n";
                }
            }
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