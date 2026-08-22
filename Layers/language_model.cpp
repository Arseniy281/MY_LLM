#include "language_model.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <errno.h>

LanguageModel::LanguageModel(size_t vocab_size, size_t embed_dim, size_t num_blocks, 
        size_t num_heads, size_t hidden_dim) : vocab_size_(vocab_size),
      embedding_(vocab_size, embed_dim),
      transformer_(num_blocks, embed_dim, num_heads, hidden_dim),
      lm_head_(embed_dim, vocab_size),
      gen_(std::random_device{}()) {}

std::shared_ptr<Tensor> LanguageModel::forward(const Tensor& tokens) {
    auto x = embedding_.forward({std::make_shared<Tensor>(tokens)});
    if (x->GetShape().size() == 2) {
        x->Reshape({1, x->GetShape()[0], x->GetShape()[1]});
    }
    x = transformer_.forward(*x);
    auto logits = lm_head_.forward(*x);
    return logits;
}

int LanguageModel::Sample(const Tensor& probs) {
    std::vector<float> probs_vec(probs.GetSize());
    for (size_t i = 0; i < probs.GetSize(); i++) {
        probs_vec[i] = probs.at(i);
    }
    std::discrete_distribution<int> dist(probs_vec.begin(), probs_vec.end());
    return dist(gen_);
}

int LanguageModel::SampleGreedy(const Tensor& probs) {
    int best_token = 0;
    float best_prob = probs.at(0);

    for (size_t i = 1; i < probs.GetSize(); ++i) {
        if (probs.at(i) > best_prob) {
            best_prob = probs.at(i);
            best_token = i;
        }
    }

    return best_token;
}


void LanguageModel::TopP(Tensor& last_logits, float top_p) {
    std::vector<std::pair<float, int>> indexed_probs;
    for (size_t i = 0; i < vocab_size_; i++) {
        indexed_probs.push_back({last_logits.at(i), i});
    }
    std::sort(indexed_probs.begin(), indexed_probs.end(), [](const auto& a, const auto& b){
            return a.first > b.first; 
        });
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
        float temperature, float top_p, int end_token_id) {

    transformer_.SetUseKVCache(true);
    transformer_.ResetCache();
    std::vector<int> generated_tokens;
    generated_tokens.reserve(max_len);
    generated_tokens.push_back(start_token);

    Tensor tokens_tensor({1, 1});
    tokens_tensor.at({0, 0}) = (float)start_token;
    auto output = forward(tokens_tensor);

    while (generated_tokens.size() < (size_t)(max_len)) {
        Tensor last_logits({vocab_size_});
        size_t last_position = output->GetShape()[1] - 1;
        for (size_t i = 0; i < vocab_size_; i++) {
            last_logits.at(i) =
                output->at({0, last_position, i});
        }

        last_logits /= temperature;

        last_logits = softmax_.forward(last_logits);

        if (top_p > 0.0f) {
            TopP(last_logits, top_p);
        }

        int next_token = SampleGreedy(last_logits);
        if (next_token == end_token_id) {
            break;
        }

        generated_tokens.push_back(next_token);
        Tensor next_tensor({1, 1});
        next_tensor.at({0, 0}) = (float)next_token;
        output = forward(next_tensor);
    }

    transformer_.SetUseKVCache(false);
    return generated_tokens;
}

void LanguageModel::SaveModel(const std::string& folder) {
    std::filesystem::create_directories(folder);
    embedding_.Save(folder + "/embedding");
    transformer_.Save(folder);
    lm_head_.Save(folder, "lm_head");
}

void LanguageModel::LoadModel(const std::string& folder) {
    embedding_.Load(folder + "/embedding");
    transformer_.Load(folder);
    lm_head_.Load(folder, "lm_head");
    transformer_.ResetCache();
}

const Tensor& LanguageModel::GetEmbeddings() const {
    return embedding_.GetEmbeddings();
}
const Tensor& LanguageModel::GetLMHeadWeights() const {
    return lm_head_.GetWeights();
}
const Tensor& LanguageModel::GetLMHeadBias() const {
    return lm_head_.GetBias();
}

void LanguageModel::ResetCache() {
    transformer_.ResetCache();
}

void LanguageModel::Update(float lr) {
    embedding_.Update(lr);
    transformer_.Update(lr);
    lm_head_.Update(lr);
}

void LanguageModel::ClearGrad() {
    embedding_.ClearGrad();
    transformer_.ClearGrad();
    lm_head_.ClearGrad();
}

void LanguageModel::ScaleGrad(float factor) {
    embedding_.ScaleGrad(factor);
    transformer_.ScaleGrad(factor);
    lm_head_.ScaleGrad(factor);
}

void LanguageModel::SetUseKVCache(bool value) {
    transformer_.SetUseKVCache(value);
}