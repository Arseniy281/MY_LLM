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

public:
    LanguageModel(size_t vocab_size, size_t embed_dim, size_t num_blocks, 
        size_t num_heads, size_t hidden_dim);
    Tensor forward(const Tensor& tokens);
    std::vector<int> generate(int start_token, int max_len = 100, 
        float temperature = 1.0f, int top_k = 0, float top_p = 0.0f, 
        int end_token_id = -1);

    int Sample(const Tensor& probs);
    void TopP(Tensor& last_logits, float top_p = 0.9);
};