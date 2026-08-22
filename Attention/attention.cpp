#include "../Layers/linear_layer.h"
#include "../Matmul/matmul.h"
#include "../Tensor/tensor.h"
#include "rope.h"
#include "attention.h"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <sys/stat.h>
#include <errno.h> 

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads) 
    : embed_dim_(embed_dim), num_heads_(num_heads) {
    head_dim_ = embed_dim_ / num_heads_;
    softmax_.resize(num_heads_);
    
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

Tensor MultiHeadAttention::GetLastToken(const Tensor& tensor) {
    std::vector<size_t> shape = tensor.GetShape();
    size_t rank = shape.size();
    size_t seq_len = shape[rank - 2];
    size_t head_dim = shape[rank - 1];
    
    std::vector<size_t> new_shape = shape;
    new_shape[rank - 2] = 1;
    
    Tensor result(new_shape);
    
    for (size_t i = 0; i < result.GetSize(); i++) {
        std::vector<size_t> coords = Tensor::IndexToCoord(i, new_shape);
        
        std::vector<size_t> src_coords = coords;
        src_coords[rank - 2] = seq_len - 1;
        
        result.at(coords) = tensor.at(src_coords);
    }
    
    return result;
}

Tensor GetLastRows(const Tensor& tensor, size_t n) {
    std::vector<size_t> shape = tensor.GetShape();
    size_t rows = shape[0];
    size_t cols = shape[1];
    
    Tensor result({n, cols});
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.at({i, j}) = tensor.at({rows - n + i, j});
        }
    }
    return result;
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(const Tensor& x) {
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

    if (!use_kv_cache_) {
        rope_start_pos_ = 0;

        for (size_t i = 0; i < num_heads_; i++) {
            Q[i] = std::make_shared<Tensor>(RoPE(*Q[i], 0));
            K[i] = std::make_shared<Tensor>(RoPE(*K[i], 0));
        }

        saved_Q_rot_ = Q;
        saved_K_rot_ = K;

        size_t seq_len = x.GetShape()[1];

        Tensor mask = CreateCausalMask(seq_len);
        mask.Reshape({1, seq_len, seq_len});

        std::vector<Tensor> head_outputs;
        head_outputs.reserve(num_heads_);
        std::vector<Tensor> attention_weights;
        attention_weights.reserve(num_heads_);

        for (size_t i = 0; i < num_heads_; i++) {
            Tensor K_transposed = K[i]->Transpose();
            Tensor scores = MatMul(*Q[i], K_transposed);
            scores /= std::sqrt((float)head_dim_);
            scores += mask;
            Tensor weights = softmax_[i].forward(scores);
            attention_weights.push_back(weights);
            Tensor head_output = MatMul(weights, *V[i]);
            head_outputs.push_back(head_output);
        }

        saved_attention_weights_ = attention_weights;
        Tensor concatenated = Tensor::Concatenate(head_outputs, 2);
        saved_concatenated_ = concatenated;
        auto output = output_layer_.forward(concatenated);
        saved_output_ = *output;
        return output;
    }

    size_t rope_start_pos = 0;

    if (!kv_cache_K_.empty()) {
        rope_start_pos = kv_cache_K_[0].GetShape()[1];
    }

    rope_start_pos_ = rope_start_pos;

    for (size_t i = 0; i < num_heads_; i++) {
        Q[i] = std::make_shared<Tensor>(RoPE(*Q[i], rope_start_pos));
        K[i] = std::make_shared<Tensor>(RoPE(*K[i], rope_start_pos));
    }

    saved_Q_rot_ = Q;
    saved_K_rot_ = K;

    size_t current_seq_len = x.GetShape()[1];

    if (is_first_token_) {
        is_first_token_ = false;

        kv_cache_K_.clear();
        kv_cache_V_.clear();

        kv_cache_K_.reserve(num_heads_);
        kv_cache_V_.reserve(num_heads_);

        for (size_t i = 0; i < num_heads_; i++) {
            kv_cache_K_.push_back(*K[i]);
            kv_cache_V_.push_back(*V[i]);
        }
    } else {
        if (current_seq_len != 1) {
            std::cerr
                << "WARNING: KV-cache decode expected seq_len=1, "
                << "but got seq_len=" << current_seq_len
                << "\n";
        }

        for (size_t i = 0; i < num_heads_; i++) {
            Tensor new_K = *K[i];
            Tensor new_V = *V[i];
            kv_cache_K_[i] = Tensor::Concatenate({kv_cache_K_[i], new_K}, 1);
            kv_cache_V_[i] = Tensor::Concatenate({kv_cache_V_[i], new_V}, 1);
        }
    }

    size_t total_len = kv_cache_K_[0].GetShape()[1];
    Tensor mask = CreateCausalMask(total_len);
    Tensor mask_for_q = GetLastRows(mask, current_seq_len);
    mask_for_q.Reshape({1, current_seq_len, total_len});

    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads_);

    std::vector<Tensor> attention_weights;
    attention_weights.reserve(num_heads_);

    for (size_t i = 0; i < num_heads_; i++) {
        Tensor K_cache = kv_cache_K_[i];
        Tensor V_cache = kv_cache_V_[i];
        Tensor K_transposed = K_cache.Transpose();
        Tensor scores = MatMul(*Q[i], K_transposed);

        scores /= std::sqrt((float)head_dim_);
        scores += mask_for_q;
        Tensor weights =softmax_[i].forward(scores);
        attention_weights.push_back(weights);

        Tensor head_output = MatMul(weights, V_cache);

        head_outputs.push_back(head_output);
    }

    saved_attention_weights_ = attention_weights;
    Tensor concatenated = Tensor::Concatenate(head_outputs, 2);
    saved_concatenated_ = concatenated;

    auto output = output_layer_.forward(concatenated);

    saved_output_ = *output;
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

    Tensor grad_x({batch, seq_len, embed_dim}, 0.0f);

    for (size_t i = 0; i < head_grads.size(); i++) {
        Tensor grad_V = MatMul(saved_attention_weights_[i].Transpose(), head_grads[i]);
        Tensor grad_scores = MatMul(head_grads[i], saved_V_[i]->Transpose());

        grad_scores = softmax_[i].backward(grad_scores);
        grad_scores /= std::sqrt((float)head_dim_);
        Tensor grad_Q = MatMul(grad_scores, *saved_K_rot_[i]);

        Tensor grad_K = MatMul(grad_scores.Transpose(), *saved_Q_rot_[i]);

        grad_Q = RoPE_backward(grad_Q, rope_start_pos_);
        grad_K = RoPE_backward(grad_K, rope_start_pos_);

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

void MultiHeadAttention::ScaleGrad(float factor) {
    for (auto& layer : q_layers_) layer.ScaleGrad(factor);
    for (auto& layer : k_layers_) layer.ScaleGrad(factor);
    for (auto& layer : v_layers_) layer.ScaleGrad(factor);
    output_layer_.ScaleGrad(factor);
}

void MultiHeadAttention::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + folder);
    }

    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_[i].Save(folder, "q_" + std::to_string(i));
        k_layers_[i].Save(folder, "k_" + std::to_string(i));
        v_layers_[i].Save(folder, "v_" + std::to_string(i));
    }
    output_layer_.Save(folder, "output");
}

void MultiHeadAttention::Load(const std::string& folder) {
    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_[i].Load(folder, "q_" + std::to_string(i));
        k_layers_[i].Load(folder, "k_" + std::to_string(i));
        v_layers_[i].Load(folder, "v_" + std::to_string(i));
    }
    output_layer_.Load(folder, "output");
}

void MultiHeadAttention::ResetCache() {
    is_first_token_ = true;
    kv_cache_K_.clear();
    kv_cache_V_.clear();
    rope_start_pos_ = 0;
}

size_t MultiHeadAttention::GetRopeStartPos() const {
    return rope_start_pos_;
}

void MultiHeadAttention::SetUseKVCache(bool value) {
    use_kv_cache_ = value;
}