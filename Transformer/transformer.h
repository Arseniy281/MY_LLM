#pragma once

#include "transformer_block.h"
#include <vector>

class Transformer {
private:
    std::vector<TransformerBlock> blocks_;
    size_t blocks_count_;
public:
    Transformer() = default;
    Transformer(size_t n, size_t embed_dim,
        size_t num_heads, size_t hidden_dim);

    std::shared_ptr<Tensor> forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);
    void Save(const std::string& folder) const;
    void Load(const std::string& folder);

    void ResetCache();
    void SetUseKVCache(bool value);
};