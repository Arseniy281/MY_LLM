#include "language_model.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>
#include <algorithm>

LanguageModel::LanguageModel(size_t vocab_size, size_t embed_dim, size_t num_blocks, 
        size_t num_heads, size_t hidden_dim) : vocab_size_(vocab_size),
      embedding_(vocab_size, embed_dim),
      transformer_(num_blocks, embed_dim, num_heads, hidden_dim),
      lm_head_(embed_dim, vocab_size) {}

Tensor LanguageModel::forward(const Tensor& tokens) {
    Tensor x = *embedding_.forward({std::make_shared<Tensor>(tokens)});
    if (x.GetShape().size() == 2) {
        x.Reshape({1, x.GetShape()[0], x.GetShape()[1]});
    }
    x = transformer_.forward(x);
    Tensor logits = *lm_head_.forward(x);
    return logits;
}

int LanguageModel::Sample(const Tensor& probs) {
    std::vector<float> probs_vec(probs.GetSize());
    for (size_t i = 0; i < probs.GetSize(); i++) {
        probs_vec[i] = probs.at(i);
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probs_vec.begin(), probs_vec.end());
    return dist(gen);
}

void LanguageModel::TopP(Tensor& last_logits, float top_p) {
    std::vector<std::pair<float, int>> indexed_probs;
    for (size_t i = 0; i < vocab_size_; i++) {
        indexed_probs.push_back({last_logits.at(i), i});
    }
    std::sort(indexed_probs.begin(), indexed_probs.end(), [](auto& a, auto& b){ return a.first > b.first; });
    float total_out = 0.0f;
    size_t border = vocab_size_ - 1;
    for (size_t i = 0; i < vocab_size_; i++) {
        total_out += indexed_probs[i].first;
        if (total_out >= top_p) { 
            border = i;
            break;
        }
    }
    for (size_t i = border + 1; i < vocab_size_; i++) {
        last_logits.at(indexed_probs[i].second) = 0.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i <= border; i++) {
        sum += indexed_probs[i].first;
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i <= border; i++) {
            last_logits.at(indexed_probs[i].second) = indexed_probs[i].first / sum;
        }
    }
}

std::vector<int> LanguageModel::generate(int start_token, int max_len, 
        float temperature, int top_k, float top_p, int end_token_id) {
    std::vector<int> generated_tokens;
    generated_tokens.push_back(start_token);

    while (generated_tokens.size() < max_len) {
        Tensor tokens_tensor({1, generated_tokens.size()});
        for (size_t i = 0; i < generated_tokens.size(); i++) {
            tokens_tensor.at({0, i}) = (float)(generated_tokens[i]);
        }

        Tensor output = forward(tokens_tensor);
        Tensor last_logits({vocab_size_});
        for (size_t i = 0; i < vocab_size_; i++) {
            last_logits.at(i) = output.at({0, output.GetShape()[1] - 1, i});
        }

        last_logits /= temperature;
        if (top_p > 0.0f) {
            TopP(last_logits, top_p);
        }
        last_logits = softmax_.forward(last_logits);
        int next_token = Sample(last_logits);

        if (next_token == end_token_id) { break; }
        generated_tokens.push_back(next_token);
    }

    return generated_tokens;
}