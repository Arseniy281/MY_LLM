#include "../Tensor/tensor.h"
#include "../Layers/linear_layer.h"
#include "softmax.h"
#include <iostream>
#include <memory>
#include <vector>

class Attention {
private:
    std::vector<LinearLayer> q_layers_;
    std::vector<LinearLayer> k_layers_;
    std::vector<LinearLayer> v_layers_;
    
    size_t embed_dim_;
    size_t head_dim_;
    size_t num_heads_;
    
    LinearLayer output_layer_;

public:
    Attention(size_t embed_dim, size_t num_heads = 1);

    Tensor CreateCausalMask(size_t seq_len);
    Tensor forward(const std::shared_ptr<Tensor>& x);
};