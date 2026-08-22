#include "../Tensor/tensor.h"
#include "../Layers/linear_layer.h"
#include "softmax.h"
#include <iostream>
#include <memory>
#include <vector>

class MultiHeadAttention {
private:
    std::vector<LinearLayer> q_layers_;
    std::vector<LinearLayer> k_layers_;
    std::vector<LinearLayer> v_layers_;
    
    size_t embed_dim_;
    size_t head_dim_;
    size_t num_heads_;

    std::vector<Softmax> softmax_;
    
    LinearLayer output_layer_;

    std::vector<std::shared_ptr<Tensor>> saved_Q_;
    std::vector<std::shared_ptr<Tensor>> saved_K_;
    std::vector<std::shared_ptr<Tensor>> saved_V_;
    std::vector<std::shared_ptr<Tensor>> saved_Q_rot_;
    std::vector<std::shared_ptr<Tensor>> saved_K_rot_;
    std::vector<Tensor> saved_attention_weights_;
    Tensor saved_concatenated_;
    Tensor saved_output_;
    Tensor saved_mask_;

    std::vector<Tensor> kv_cache_K_;
    std::vector<Tensor> kv_cache_V_;
    bool use_kv_cache_ = false;
    bool is_first_token_ = true;

    size_t rope_start_pos_ = 0;

    Tensor GetLastToken(const Tensor& tensor);
public:
    MultiHeadAttention(size_t embed_dim, size_t num_heads = 1);

    Tensor CreateCausalMask(size_t seq_len);
    std::shared_ptr<Tensor> forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& folder) const;
    void Load(const std::string& folder);

    void ResetCache();
    size_t GetRopeStartPos() const;
    void SetUseKVCache(bool value);
};