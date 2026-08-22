#include "../Layers/language_model.h"
#include "../Tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

void TestFullVsKVCache(
    LanguageModel& model,
    const std::vector<int>& tokens
) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " FULL FORWARD vs KV-CACHE FORWARD\n";
    std::cout << "========================================\n";

    if (tokens.empty()) {
        std::cout << "ERROR: empty token sequence\n";
        return;
    }

    // ------------------------------------------------------------
    // 1. FULL FORWARD
    //
    // Каждый prefix прогоняем целиком:
    //
    // [7]
    // [7 4]
    // [7 4 11]
    // ...
    //
    // Это reference implementation.
    // ------------------------------------------------------------

    std::vector<std::vector<float>> full_logits;

    for (size_t len = 1; len <= tokens.size(); len++) {
        model.ResetCache();

        Tensor input({1, len});

        for (size_t i = 0; i < len; i++) {
            input.at({0, i}) =
                static_cast<float>(tokens[i]);
        }

        auto output = model.forward(input);

        size_t vocab_size = output->GetShape()[2];
        size_t last_pos = len - 1;

        std::vector<float> logits(vocab_size);

        for (size_t v = 0; v < vocab_size; v++) {
            logits[v] =
                output->at({0, last_pos, v});
        }

        full_logits.push_back(logits);
    }

    // ------------------------------------------------------------
    // 2. KV-CACHE FORWARD
    //
    // Один ResetCache().
    //
    // Потом:
    //
    // forward(7)
    // forward(4)
    // forward(11)
    // ...
    //
    // После каждого forward берём последний token.
    // ------------------------------------------------------------

    model.ResetCache();

    std::vector<std::vector<float>> cache_logits;

    for (size_t i = 0; i < tokens.size(); i++) {
        Tensor input({1, 1});

        input.at({0, 0}) =
            static_cast<float>(tokens[i]);

        auto output = model.forward(input);

        size_t vocab_size = output->GetShape()[2];

        std::vector<float> logits(vocab_size);

        for (size_t v = 0; v < vocab_size; v++) {
            logits[v] =
                output->at({0, 0, v});
        }

        cache_logits.push_back(logits);
    }

    // ------------------------------------------------------------
    // 3. COMPARE
    // ------------------------------------------------------------

    size_t vocab_size = full_logits[0].size();

    float global_max_diff = 0.0f;
    size_t global_position = 0;
    size_t global_token = 0;

    size_t different_predictions = 0;

    for (size_t pos = 0; pos < tokens.size(); pos++) {
        float max_diff = 0.0f;
        size_t max_diff_token = 0;

        for (size_t v = 0; v < vocab_size; v++) {
            float diff =
                std::abs(
                    full_logits[pos][v] -
                    cache_logits[pos][v]
                );

            if (diff > max_diff) {
                max_diff = diff;
                max_diff_token = v;
            }

            if (diff > global_max_diff) {
                global_max_diff = diff;
                global_position = pos;
                global_token = v;
            }
        }

        // --------------------------------------------------------
        // argmax FULL
        // --------------------------------------------------------

        size_t full_prediction = 0;

        for (size_t v = 1; v < vocab_size; v++) {
            if (full_logits[pos][v] >
                full_logits[pos][full_prediction]) {
                full_prediction = v;
            }
        }

        // --------------------------------------------------------
        // argmax CACHE
        // --------------------------------------------------------

        size_t cache_prediction = 0;

        for (size_t v = 1; v < vocab_size; v++) {
            if (cache_logits[pos][v] >
                cache_logits[pos][cache_prediction]) {
                cache_prediction = v;
            }
        }

        bool same_prediction =
            full_prediction == cache_prediction;

        if (!same_prediction) {
            different_predictions++;
        }

        std::cout
            << "Position " << pos
            << " | input=" << tokens[pos]
            << " | FULL=" << full_prediction
            << " | CACHE=" << cache_prediction
            << " | max_diff=" << max_diff;

        if (same_prediction) {
            std::cout << "  OK";
        } else {
            std::cout << "  !!! DIFFERENT !!!";
        }

        std::cout << "\n";

        // Если хочется увидеть конкретный токен,
        // который сильнее всего отличается:

        if (max_diff > 1e-4f) {
            std::cout
                << "    largest diff at vocab token "
                << max_diff_token
                << "\n";

            std::cout
                << "    FULL  = "
                << full_logits[pos][max_diff_token]
                << "\n";

            std::cout
                << "    CACHE = "
                << cache_logits[pos][max_diff_token]
                << "\n";
        }
    }

    // ------------------------------------------------------------
    // 4. FINAL RESULT
    // ------------------------------------------------------------

    std::cout << "\n";
    std::cout << "----------------------------------------\n";
    std::cout
        << "Different predictions: "
        << different_predictions
        << " / "
        << tokens.size()
        << "\n";

    std::cout
        << "Global max diff: "
        << global_max_diff
        << "\n";

    std::cout
        << "Global max diff position: "
        << global_position
        << "\n";

    std::cout
        << "Global max diff vocab token: "
        << global_token
        << "\n";

    if (different_predictions == 0 &&
        global_max_diff < 1e-4f) {

        std::cout << "\n";
        std::cout
            << "RESULT: FULL and KV-CACHE "
            << "are equivalent.\n";
    }
    else if (different_predictions == 0) {

        std::cout << "\n";
        std::cout
            << "RESULT: predictions match, "
            << "but logits have numerical differences.\n";
    }
    else {

        std::cout << "\n";
        std::cout
            << "RESULT: KV-CACHE IS NOT EQUIVALENT!\n";
    }

    std::cout << "========================================\n";
}

// --- Вспомогательные функции ---

#include "../Tensor/tensor.h"

void TestPredictions(
    LanguageModel& model,
    const std::vector<int>& tokens
) {
    std::cout << "\n=== Prediction Test ===\n";

    for (size_t i = 0; i < tokens.size() - 1; i++) {
        model.ResetCache();

        Tensor input({1, i + 1});

        for (size_t j = 0; j <= i; j++) {
            input.at({0, j}) = static_cast<float>(tokens[j]);
        }

        auto logits_ptr = model.forward(input);
        Tensor& logits = *logits_ptr;

        size_t vocab_size = logits.GetShape()[2];
        size_t pos = i;

        int predicted = 0;
        float best = logits.at({0, pos, 0});

        for (size_t j = 1; j < vocab_size; j++) {
            float value = logits.at({0, pos, j});

            if (value > best) {
                best = value;
                predicted = static_cast<int>(j);
            }
        }

        std::cout
            << "Step " << i
            << ": expected " << tokens[i + 1]
            << ", predicted " << predicted;

        if (predicted == tokens[i + 1]) {
            std::cout << "  ✅\n";
        } else {
            std::cout << "  ❌\n";
        }
    }

    model.ResetCache();
}

class CrossEntropyLoss {
private:
    Tensor logits_;
    size_t pos_;
    int target_;
    std::vector<float> softmax_output_;

public:
    // forward: принимает логиты [batch, seq_len, vocab_size], позицию pos и target
    Tensor forward(const Tensor& logits, size_t pos, int target) {
        logits_ = logits;
        pos_ = pos;
        target_ = target;

        size_t vocab_size = logits.GetShape()[2];

        // 1. Извлекаем последний логит (по позиции pos)
        Tensor last_logits({vocab_size});
        for (size_t i = 0; i < vocab_size; i++) {
            last_logits.at(i) = logits.at({0, pos, i});
        }

        const float* data = last_logits.RawData();

        // 2. Численная стабильность: вычитаем максимум
        float max_val = data[0];
        for (size_t i = 1; i < vocab_size; i++) {
            if (data[i] > max_val) max_val = data[i];
        }

        // 3. exp(x - max) и сумма
        std::vector<float> exp_vals(vocab_size);
        float sum_exp = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            exp_vals[i] = std::exp(data[i] - max_val);
            sum_exp += exp_vals[i];
        }

        // 4. Softmax
        softmax_output_.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; i++) {
            softmax_output_[i] = exp_vals[i] / sum_exp;
        }

        // 5. Loss = -log(softmax[target])
        float loss_value = -std::log(softmax_output_[target]);
        return Tensor({1, 1}, loss_value);
    }

    // backward: возвращает градиент для logits (3D)
    Tensor backward() {
        size_t vocab_size = logits_.GetShape()[2];

        // Градиент для last_logits (softmax - 1 для target)
        Tensor grad_last_logits({vocab_size});
        float* grad_data = grad_last_logits.RawData();

        for (size_t i = 0; i < vocab_size; i++) {
            grad_data[i] = softmax_output_[i];
        }
        grad_data[target_] -= 1.0f;

        // Создаём градиент для всего logits (3D)
        Tensor grad_logits(logits_.GetShape());
        float* grad_logits_data = grad_logits.RawData();

        // Заполняем нулями
        for (size_t i = 0; i < grad_logits.GetSize(); i++) {
            grad_logits_data[i] = 0.0f;
        }

        // Копируем градиент в нужную позицию (pos)
        for (size_t i = 0; i < vocab_size; i++) {
            grad_logits.at({0, pos_, i}) = grad_data[i];
        }

        return grad_logits;
    }
};

std::vector<int> PrepareBatch(const std::string& text, BPETokenizer& tokenizer) {
    std::vector<size_t> ids = tokenizer.Encode(text);
    std::vector<int> tokens;
    for (size_t id : ids) {
        tokens.push_back(static_cast<int>(id));
    }
    return tokens;
}

void PrintTokens(const std::vector<int>& tokens, BPETokenizer& tokenizer) {
    std::vector<size_t> ids;
    for (int t : tokens) ids.push_back(static_cast<size_t>(t));
    std::cout << tokenizer.Decode(ids) << std::endl;
}

// --- Главный тест ---

int main() {
    std::cout << "=== Training & Generation Test ===\n\n";
    
    // 1. Токенизатор
    BPETokenizer tokenizer;
    
    // 2. Обучающие данные (маленький текст)
    std::string corpus = "hello world hello world hello world";
    tokenizer.Train(corpus, 50);
    tokenizer.Save("vocab.txt");
    std::cout << "Vocabulary size: " << tokenizer.GetVocabSize() << "\n\n";
    
    // 3. Создаём модель
    size_t vocab_size = tokenizer.GetVocabSize();
    size_t embed_dim = 16;
    size_t num_blocks = 2;
    size_t num_heads = 2;
    size_t hidden_dim = 32;
    float learning_rate = 0.01f;
    int epochs = 1000;
    
    LanguageModel model(vocab_size, embed_dim, num_blocks, num_heads, hidden_dim);
    std::cout << "Model created\n\n";
    
    // 4. Подготовка данных
    std::vector<int> tokens = PrepareBatch("hello world", tokenizer);
    std::cout << "Tokens: ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << "\n\n";
    
    // 5. Цикл обучения
    std::cout << "Training started...\n";
    CrossEntropyLoss loss_fn;

    model.SetUseKVCache(false);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < tokens.size() - 1; i++) {
            model.ResetCache();

            std::vector<int> prefix(
                tokens.begin(),
                tokens.begin() + i + 1
            );

            Tensor input({1, prefix.size()});

            for (size_t j = 0; j < prefix.size(); j++) {
                input.at({0, j}) =
                    static_cast<float>(prefix[j]);
            }

            auto logits_ptr = model.forward(input);
            Tensor& logits = *logits_ptr;

            Tensor loss = loss_fn.forward(
                logits,
                prefix.size() - 1,
                tokens[i + 1]
            );

            total_loss += loss.at(0);

            Tensor grad_logits = loss_fn.backward();
            logits.backward(grad_logits);

            model.Update(learning_rate);
            model.ClearGrad();
        }

        std::cout
            << "Epoch " << epoch
            << ", Loss: "
            << total_loss / (tokens.size() - 1)
            << "\n";
    }

    TestPredictions(model, tokens);

    std::vector<int> test_tokens = {
        7, 4, 11, 11, 14, 22, 14, 17, 11, 3
    };

    TestFullVsKVCache(model, test_tokens);
    
    // 6. Генерация
    std::cout << "\n=== Generation ===\n";
    
    // Сброс кэша
    model.ResetCache();
    
    int start_token = tokens[0];
    int end_token_id = tokenizer.GetTokenId("<UNK>"); // или -1
    int max_len = 20;
    // float temperature = 0.8f;
    // float top_p = 0.9f;
    float temperature = 1.0f;
    float top_p = 0.0f;

    // Обучим модель (здесь просто генерация без обучения)
    std::cout << "Generating with greedy sample:\n";
    std::vector<int> generated = model.generate(start_token, max_len, temperature, top_p, end_token_id);
    
    std::cout << "Tokens: ";
    for (int t : generated) std::cout << t << " ";
    std::cout << "\n";
    
    std::cout << "Text: ";
    PrintTokens(generated, tokenizer);
    std::cout << "\n";
    
    std::cout << "=== Test completed ===\n";
    return 0;
}