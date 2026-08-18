#include "../Layers/linear_layer.h"
#include "../Matmul/matmul.h"
#include "../Tensor/tensor.h"
#include "rope.h"
#include "attention.h"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads) 
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

Tensor MultiHeadAttention::CreateCausalMask(size_t seq_len) {
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

Tensor MultiHeadAttention::forward(const Tensor& x) {
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
    saved_Q_ = Q;
    saved_K_ = K;
    saved_V_ = V;
    for (size_t i = 0; i < num_heads_; i++) {
        Q[i] = std::make_shared<Tensor>(RoPE(*Q[i]));
        K[i] = std::make_shared<Tensor>(RoPE(*K[i]));
    }
    saved_Q_rot_ = Q;
    saved_K_rot_ = K;

    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads_);
    Tensor mask = CreateCausalMask(x.GetShape()[1]);
    std::vector<Tensor> attention_weights;
    attention_weights.reserve(num_heads_);
    for (size_t i = 0; i < num_heads_; i++) {
        Tensor scores = MatMul(*Q[i], K[i]->Transpose());
        scores /= std::sqrt(head_dim_);
        scores += mask;
        Tensor weights = softmax_.forward(scores);
        attention_weights.push_back(weights);
        head_outputs.push_back(MatMul(weights, *V[i]));
    }

    saved_attention_weights_ = attention_weights;

    Tensor concatenated = Tensor::Concatenate(head_outputs, 2);
    saved_concatenated_ = concatenated;

    Tensor output = *output_layer_.forward(concatenated);
    saved_output_ = output;
    return output;
}

Tensor MultiHeadAttention::backward(const Tensor& grad_output) {
    Tensor grad_concatenated = output_layer_.backward(grad_output);

    std::vector<Tensor> head_grads;
    size_t batch = grad_concatenated.GetShape()[0];
    size_t seq_len = grad_concatenated.GetShape()[1];
    size_t embed_dim = grad_concatenated.GetShape()[2];
    size_t head_dim = embed_dim / num_heads_;
    
    for (size_t h = 0; h < num_heads_; h++) {
        Tensor head_grad({batch, seq_len, head_dim});
        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < head_dim; d++) {
                    head_grad.at({b, s, d}) = grad_concatenated.at({b, s, h * head_dim + d});
                }
            }
        }
        head_grads.push_back(head_grad);
    }

    Tensor grad_x = Tensor({batch, seq_len, embed_dim}, 0.0f);

    for (size_t i = 0; i < head_grads.size(); i++) {
        Tensor grad_V = MatMul(saved_attention_weights_[i].Transpose(), head_grads[i]);
        Tensor grad_scores = MatMul(head_grads[i], saved_V_[i]->Transpose());
        grad_scores = softmax_.backward(grad_scores);
        grad_scores /= sqrt(head_dim_);
        Tensor grad_Q = MatMul(grad_scores, *saved_K_rot_[i]);
        Tensor grad_K = MatMul(saved_Q_rot_[i]->Transpose(), grad_scores);
        grad_Q = RoPE_backward(grad_Q);
        grad_K = RoPE_backward(grad_K);
        grad_x += q_layers_[i].backward(grad_Q);
        grad_x += k_layers_[i].backward(grad_K);
        grad_x += v_layers_[i].backward(grad_V);
    }

    return grad_x;
}

void MultiHeadAttention::Update(float lr) {
    for (auto& layer : q_layers_) layer.Update(lr);
    for (auto& layer : k_layers_) layer.Update(lr);
    for (auto& layer : v_layers_) layer.Update(lr);
    output_layer_.Update(lr);
}

void MultiHeadAttention::ClearGrad() {
    for (auto& layer : q_layers_) layer.ClearGrad();
    for (auto& layer : k_layers_) layer.ClearGrad();
    for (auto& layer : v_layers_) layer.ClearGrad();
    output_layer_.ClearGrad();
}