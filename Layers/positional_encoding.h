#include<../Tensor/tensor.h>
#include <iostream>
#include <cmath>

class PositionalEncoding {
public:
    Tensor forward(size_t seq_len, size_t embed_dim) {
        Tensor tensor({seq_len, embed_dim});
        for (size_t pos = 0; pos < seq_len; pos++) {
            for (size_t i = 0; i < embed_dim; i++) {
                float angle = (float)(pos) / std::pow(10000.0f, (2.0f * i) / embed_dim);
                if (i % 2 == 0) {
                    tensor.at({pos, i}) = std::sin(angle);
                } else {
                    tensor.at({pos, i}) = std::cos(angle);
                }
            }
        }
        return tensor;
    }
};