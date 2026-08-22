#include "../Transformer/transformer.h"
#include "../Autograd/embedding_layer.h"
#include "../Tensor/tensor.h"
#include "linear_layer.h"
#include <vector>

class LanguageModel {
private:
    EmbeddingLayer embedding_;
    Transformer transformer_;
    LinearLayer lm_head_;
    Softmax softmax_;
    size_t vocab_size_;

    std::mt19937 gen_;

public:
    LanguageModel(size_t vocab_size, size_t embed_dim, size_t num_blocks, 
        size_t num_heads, size_t hidden_dim);
    std::shared_ptr<Tensor> forward(const Tensor& tokens);
    std::vector<int> generate(int start_token, int max_len = 100, 
        float temperature = 1.0f, float top_p = 0.0f, 
        int end_token_id = -1);

    int Sample(const Tensor& probs);
    int SampleGreedy(const Tensor& probs);
    void TopP(Tensor& last_logits, float top_p = 0.9);

    void SaveModel(const std::string& folder);
    void LoadModel(const std::string& folder);

    const Tensor& GetEmbeddings() const;
    const Tensor& GetLMHeadWeights() const;
    const Tensor& GetLMHeadBias() const;

    void ResetCache();
    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);
    void SetUseKVCache(bool value);
};