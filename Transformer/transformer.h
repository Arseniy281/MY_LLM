#pragma once

#include "transformer_block.h"
#include <vector>

class Transformer {
private:
    std::vector<TransformerBlock> blocks_;
    size_t blocks_count_;
public:
    Transformer(size_t n, size_t embed_dim,
        size_t num_heads, size_t hidden_dim);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    void Update(float lr);
    void ClearGrad();
};