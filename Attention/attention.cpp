#include "../Layers/linear_layer.h"
#include "../Matmul/matmul.h"
#include "../Tensor/tensor.h"
#include "rope.h"
#include "attention.h"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

Attention::Attention(size_t embed_dim, size_t num_heads) 
    : embed_dim_(embed_dim), num_heads_(num_heads) {
    head_dim_ = embed_dim_ / num_heads_;
    
    q_layers_.reserve(num_heads_);
    k_layers_.reserve(num_heads_);
    v_layers_.reserve(num_heads_);
    
    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_.emplace_back(embed_dim_, head_dim_);
        k_layers_.emplace_back(embed_dim_, head_dim_);
        v_layers_.emplace_back(embed_dim_, head_dim_);
    }

    output_layer_ = LinearLayer(embed_dim_, embed_dim_);
}

Tensor Attention::CreateCausalMask(size_t seq_len) {
    Tensor mask({seq_len, seq_len});
    float* data = mask.RawData();
    float neg_inf = -1e9f;
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            if (j <= i) {
                data[i * seq_len + j] = 0.0f;
            } else {
                data[i * seq_len + j] = neg_inf;
            }
        }
    }
    return mask;
}

Tensor Attention::forward(const std::shared_ptr<Tensor>& x) {
    std::vector<std::shared_ptr<Tensor>> Q;
    std::vector<std::shared_ptr<Tensor>> K;
    std::vector<std::shared_ptr<Tensor>> V;
    Q.reserve(num_heads_);
    K.reserve(num_heads_);
    V.reserve(num_heads_);
    for (size_t i = 0; i < num_heads_; i++) {
        Q.emplace_back(q_layers_[i].forward(x));
        K.emplace_back(k_layers_[i].forward(x));
        V.emplace_back(v_layers_[i].forward(x));
    }
    for (size_t i = 0; i < num_heads_; i++) {
        Q[i] = std::make_shared<Tensor>(RoPE(*Q[i]));
        K[i] = std::make_shared<Tensor>(RoPE(*K[i]));
    }
    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads_);
    Tensor mask = CreateCausalMask(x->GetShape()[1]);
    Softmax softmax;
    for (size_t i = 0; i < num_heads_; i++) {
        Tensor temp = MatMul(*Q[i], K[i]->Transpose());
        head_outputs.push_back(std::move(temp));
        head_outputs[i] /= std::sqrt(head_dim_);
        head_outputs[i] += mask;
        head_outputs[i] = softmax.forward(head_outputs[i]);
        head_outputs[i] = MatMul(head_outputs[i], *V[i]);
    }
    Tensor concatenated = Tensor::Concatenate(head_outputs, 2);

    Tensor output = *output_layer_.forward(std::make_shared<Tensor>(concatenated));
    return output;
}