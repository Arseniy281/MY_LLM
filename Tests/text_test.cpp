#include "../Layers/language_model.h"
#include "../Tokenizer/bpe_tokenizer.h"
#include "../Tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

static std::unordered_map<int, std::string> ID2STR;

static const float LEARNING_RATE = 0.001f;
static const int EPOCHS = 3000;

static const size_t VOCAB_SIZE = 100;
static const size_t EMBED_DIM = 32;
static const size_t NUM_BLOCKS = 2;
static const size_t NUM_HEADS = 2;
static const size_t HIDDEN_DIM = 64;

static const size_t MAX_GENERATION_TOKENS = 3;
static const float TRAIN_ACCURACY_THRESHOLD = 0.90f;
static const std::string SAVE_FOLDER = "trained_model";

class CrossEntropyLoss {
private:
    Tensor logits_;

    size_t position_ = 0;
    int target_ = 0;

    std::vector<float> probabilities_;

public:

    Tensor Forward(
        const Tensor& logits,
        size_t position,
        int target
    ) {
        logits_ = logits;
        position_ = position;
        target_ = target;

        size_t vocab_size =
            logits.GetShape()[2];

        Tensor current_logits({
            vocab_size
        });

        for (size_t i = 0;
             i < vocab_size;
             i++) {

            current_logits.at(i) =
                logits.at({
                    0,
                    position,
                    i
                });
        }

        float max_value =
            current_logits.at(0);

        for (size_t i = 1;
             i < vocab_size;
             i++) {

            max_value =
                std::max(
                    max_value,
                    current_logits.at(i)
                );
        }

        probabilities_.resize(
            vocab_size
        );

        float sum = 0.0f;

        for (size_t i = 0;
             i < vocab_size;
             i++) {

            probabilities_[i] =
                std::exp(
                    current_logits.at(i) -
                    max_value
                );

            sum += probabilities_[i];
        }

        for (size_t i = 0;
             i < vocab_size;
             i++) {

            probabilities_[i] /= sum;
        }

        float probability =
            std::max(
                probabilities_[target_],
                1e-12f
            );

        return Tensor(
            {1, 1},
            -std::log(probability)
        );
    }

    Tensor Backward() {

        Tensor gradient(
            logits_.GetShape(),
            0.0f
        );

        size_t vocab_size =
            logits_.GetShape()[2];

        for (size_t i = 0;
             i < vocab_size;
             i++) {

            gradient.at({
                0,
                position_,
                i
            }) =
                probabilities_[i];
        }

        gradient.at({
            0,
            position_,
            static_cast<size_t>(
                target_
            )
        }) -= 1.0f;

        return gradient;
    }
};

struct Sentence {
    std::string text;
    std::vector<int> tokens;
};

Tensor MakeInput(
    const std::vector<int>& tokens
) {
    Tensor input({
        1,
        tokens.size()
    });

    for (size_t i = 0;
         i < tokens.size();
         i++) {

        input.at({
            0,
            i
        }) =
            static_cast<float>(
                tokens[i]
            );
    }

    return input;
}

int Argmax(
    const Tensor& logits,
    size_t position
) {
    size_t vocab_size =
        logits.GetShape()[2];

    int best_token = 0;

    float best_value =
        logits.at({
            0,
            position,
            0
        });

    for (size_t i = 1;
         i < vocab_size;
         i++) {

        float value =
            logits.at({
                0,
                position,
                i
            });

        if (value > best_value) {

            best_value = value;

            best_token =
                static_cast<int>(i);
        }
    }

    return best_token;
}

float SentenceLoss(
    LanguageModel& model,
    const std::vector<int>& tokens
) {
    if (tokens.size() < 2) {
        return 0.0f;
    }

    model.ResetCache();
    model.SetUseKVCache(false);

    Tensor input =
        MakeInput(tokens);

    auto output =
        model.forward(input);

    float total_loss = 0.0f;

    size_t count = 0;

    for (size_t position = 0;
         position + 1 < tokens.size();
         position++) {

        CrossEntropyLoss loss;

        Tensor value =
            loss.Forward(
                *output,
                position,
                tokens[position + 1]
            );

        total_loss +=
            value.at(0);

        count++;
    }

    model.ResetCache();

    return total_loss /
           static_cast<float>(count);
}

float TrainSentence(
    LanguageModel& model,
    const std::vector<int>& tokens
) {
    if (tokens.size() < 2) {
        return 0.0f;
    }

    model.ResetCache();
    model.SetUseKVCache(false);

    Tensor input =
        MakeInput(tokens);

    auto output =
        model.forward(input);

    float total_loss = 0.0f;

    size_t count = 0;

    Tensor total_gradient(
        output->GetShape(),
        0.0f
    );

    for (size_t position = 0;
         position + 1 < tokens.size();
         position++) {

        CrossEntropyLoss loss;

        Tensor loss_value =
            loss.Forward(
                *output,
                position,
                tokens[position + 1]
            );

        total_loss +=
            loss_value.at(0);

        count++;

        Tensor gradient =
            loss.Backward();

        total_gradient +=
            gradient;
    }

    total_gradient /=
        static_cast<float>(count);

    output->backward(
        total_gradient
    );

    model.Update(
        LEARNING_RATE
    );

    model.ClearGrad();

    model.ResetCache();

    return total_loss /
           static_cast<float>(count);
}

float DatasetLoss(
    LanguageModel& model,
    const std::vector<Sentence>& dataset
) {
    float total_loss = 0.0f;

    for (const Sentence& sentence : dataset) {

        total_loss +=
            SentenceLoss(
                model,
                sentence.tokens
            );
    }

    return total_loss /
           static_cast<float>(
               dataset.size()
           );
}

float CalculateAccuracy(
    LanguageModel& model,
    const std::vector<Sentence>& dataset
) {
    size_t correct = 0;
    size_t total = 0;

    model.SetUseKVCache(false);

    for (const Sentence& sentence : dataset) {

        if (sentence.tokens.size() < 2) {
            continue;
        }

        model.ResetCache();

        Tensor input =
            MakeInput(sentence.tokens);

        auto output =
            model.forward(input);

        for (size_t position = 0;
             position + 1 < sentence.tokens.size();
             position++) {

            int prediction =
                Argmax(
                    *output,
                    position
                );

            int target =
                sentence.tokens[position + 1];

            if (prediction == target) {
                correct++;
            }

            total++;
        }
    }

    model.ResetCache();

    if (total == 0) {
        return 0.0f;
    }

    return static_cast<float>(correct) /
           static_cast<float>(total);
}


void PrintPredictions(
    LanguageModel& model,
    const std::vector<Sentence>& dataset
) {
    std::cout
        << "\n============================================================\n"
        << "                 TRAINING PREDICTIONS\n"
        << "============================================================\n";

    model.SetUseKVCache(false);

    for (const Sentence& sentence : dataset) {

        model.ResetCache();

        Tensor input =
            MakeInput(sentence.tokens);

        auto output =
            model.forward(input);

        std::cout
            << "\n"
            << sentence.text
            << "\n";

        for (size_t position = 0;
             position + 1 < sentence.tokens.size();
             position++) {

            int prediction =
                Argmax(
                    *output,
                    position
                );

            int target =
                sentence.tokens[position + 1];

            auto token_str = [&](int id)->std::string{
                auto it = ID2STR.find(id);
                if (it != ID2STR.end()) return it->second;
                return std::to_string(id);
            };

            std::cout
                << "  "
                << token_str(sentence.tokens[position])
                << " -> "
                << token_str(prediction)
                << " target="
                << token_str(target)
                << (prediction == target
                    ? " [OK]"
                    : " [WRONG]")
                << "\n";
        }
    }

    model.ResetCache();
}

std::vector<int> Generate(
    LanguageModel& model,
    const std::vector<int>& prompt,
    size_t max_new_tokens
) {
    std::vector<int> result =
        prompt;

    model.ResetCache();
    model.SetUseKVCache(true);

    int next_token = -1;

    for (size_t i = 0;
         i < prompt.size();
         i++) {

        Tensor input({
            1,
            1
        });

        input.at({
            0,
            0
        }) =
            static_cast<float>(
                prompt[i]
            );

        auto output =
            model.forward(input);

        next_token =
            Argmax(
                *output,
                0
            );
    }

    for (size_t step = 0;
         step < max_new_tokens;
         step++) {

        if (next_token < 0) {
            break;
        }

        result.push_back(
            next_token
        );

        Tensor input({
            1,
            1
        });

        input.at({
            0,
            0
        }) =
            static_cast<float>(
                next_token
            );

        auto output =
            model.forward(input);

        next_token =
            Argmax(
                *output,
                0
            );
    }

    model.ResetCache();
    model.SetUseKVCache(false);

    return result;
}

std::vector<std::string> SplitToTokenStrings(const std::string& text) {
    std::vector<std::string> parts;
    if (text.empty()) return parts;

    std::string cur;
    bool cur_is_space = std::isspace(static_cast<unsigned char>(text[0]));

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        bool is_space = std::isspace(static_cast<unsigned char>(c));

        if (cur.empty()) {
            cur.push_back(c);
            cur_is_space = is_space;
        } else if (is_space == cur_is_space) {
            cur.push_back(c);
        } else {
            parts.push_back(cur);
            cur.clear();
            cur.push_back(c);
            cur_is_space = is_space;
        }
    }

    if (!cur.empty()) parts.push_back(cur);
    return parts;
}

void PrintTokens(
    const std::vector<int>& tokens
) {
    for (size_t i = 0; i < tokens.size(); i++) {
        int id = tokens[i];
        auto it = ID2STR.find(id);
        if (it != ID2STR.end()) {
            std::cout << it->second;
        } else {
            std::cout << id;
        }
    }
}


void TestGeneration(
    LanguageModel& model,
    const std::vector<Sentence>& dataset
) {
    std::cout
        << "\n============================================================\n"
        << "                    GENERATION\n"
        << "============================================================\n";

    std::vector<std::vector<int>> prompts;

    for (const Sentence& sentence : dataset) {

        if (sentence.tokens.size() >= 2) {

            std::vector<int> prompt;

            prompt.push_back(
                sentence.tokens[0]
            );

            if (sentence.tokens.size() >= 3) {

                prompt.push_back(
                    sentence.tokens[1]
                );
            }

            prompts.push_back(
                prompt
            );
        }
    }

    for (size_t i = 0;
         i < prompts.size();
         i++) {

        std::cout
            << "\nPrompt: ";

        PrintTokens(
            prompts[i]
        );

        std::cout
            << "\n";

        std::vector<int> generated =
            Generate(
                model,
                prompts[i],
                MAX_GENERATION_TOKENS
            );

        std::cout
            << "Generated: ";

        PrintTokens(
            generated
        );

        std::cout
            << "\n";
    }
}

bool TestKVGeneration(
    LanguageModel& model,
    const Sentence& sentence
) {
    std::cout
        << "\n============================================================\n"
        << "                  KV CACHE GENERATION\n"
        << "============================================================\n";

    if (sentence.tokens.size() < 3) {
        return false;
    }

    std::vector<int> prompt = {
        sentence.tokens[0],
        sentence.tokens[1]
    };

    std::cout
        << "Prompt: ";

    PrintTokens(prompt);

    std::cout
        << "\n";

    model.ResetCache();
    model.SetUseKVCache(false);

    Tensor full_input =
        MakeInput(prompt);

    auto full_output =
        model.forward(full_input);

    int full_prediction =
        Argmax(
            *full_output,
            prompt.size() - 1
        );

    model.ResetCache();
    model.SetUseKVCache(true);

    int cached_prediction = -1;

    for (size_t i = 0;
         i < prompt.size();
         i++) {

        Tensor one({
            1,
            1
        });

        one.at({
            0,
            0
        }) =
            static_cast<float>(
                prompt[i]
            );

        auto output =
            model.forward(one);

        if (i == prompt.size() - 1) {

            cached_prediction =
                Argmax(
                    *output,
                    0
                );
        }
    }

    model.ResetCache();
    model.SetUseKVCache(false);

    auto token_str = [&](int id)->std::string{
        auto it = ID2STR.find(id);
        if (it != ID2STR.end()) return it->second;
        return std::to_string(id);
    };

    std::cout
        << "Full forward prediction:   "
        << token_str(full_prediction)
        << "\n";

    std::cout
        << "KV-cache prediction:       "
        << token_str(cached_prediction)
        << "\n";

    bool passed =
        full_prediction ==
        cached_prediction;

    std::cout
        << (passed
            ? "[PASS] KV-cache generation\n"
            : "[FAIL] KV-cache generation\n");

    return passed;
}

bool TestSaveLoad(
    LanguageModel& model,
    const Sentence& sentence
) {
    std::cout
        << "\n============================================================\n"
        << "                    SAVE / LOAD\n"
        << "============================================================\n";

    model.SetUseKVCache(false);
    model.ResetCache();

    Tensor input =
        MakeInput(sentence.tokens);

    auto before =
        model.forward(input);


    model.SaveModel(
        SAVE_FOLDER
    );

    LanguageModel loaded_model(
        VOCAB_SIZE,
        EMBED_DIM,
        NUM_BLOCKS,
        NUM_HEADS,
        HIDDEN_DIM
    );

    loaded_model.LoadModel(
        SAVE_FOLDER
    );

    loaded_model.ResetCache();
    loaded_model.SetUseKVCache(false);

    auto after = loaded_model.forward(input);
    float max_difference = 0.0f;

    for (size_t i = 0;
         i < before->GetSize();
         i++) {

        float difference =
            std::abs(
                before->at(i) -
                after->at(i)
            );

        max_difference =
            std::max(
                max_difference,
                difference
            );
    }

    std::cout
        << "Maximum logit difference: "
        << std::scientific
        << max_difference
        << "\n";

    bool passed =
        max_difference < 1e-5f;

    std::cout
        << (passed
            ? "[PASS] Save / Load\n"
            : "[FAIL] Save / Load\n");

    loaded_model.ResetCache();

    return passed;
}

int main() {

    std::cout
        << "\n============================================================\n"
        << "             LANGUAGE MODEL TRAINING TEST\n"
        << "============================================================\n";

    std::vector<Sentence> dataset = {

        {
            "the cat eats fish",
            {79, 74, 89, 74, 94, 74, 96}
        },

        {
            "the cat eats meat",
            {79, 74, 89, 74, 94, 74, 95}
        },

        {
            "the cat likes fish",
            {79, 74, 90, 74, 94, 74, 96}
        },

        {
            "the cat likes milk",
            {79, 74, 90, 74, 93, 74, 97}
        },

        {
            "the dog eats fish",
            {79, 75, 89, 74, 94, 74, 96}
        },

        {
            "the dog eats meat",
            {79, 75, 89, 74, 94, 74, 95}
        },

        {
            "the dog likes fish",
            {79, 75, 90, 74, 94, 74, 96}
        },

        {
            "the dog likes milk",
            {79, 75, 90, 74, 93, 74, 97}
        },

        {
            "fish is food",
            {96, 74, 91, 74, 92}
        },

        {
            "meat is food",
            {95, 74, 91, 74, 92}
        },

        {
            "milk is food",
            {97, 74, 91, 74, 92}
        },

        {
            "cat is animal",
            {74, 74, 91, 74, 98}
        },

        {
            "dog is animal",
            {75, 74, 91, 74, 98}
        },

        {
            "fish is animal",
            {96, 74, 91, 74, 98}
        }
    };
    for (const Sentence& s : dataset) {
        auto parts = SplitToTokenStrings(s.text);
        if (parts.size() == s.tokens.size()) {
            for (size_t j = 0; j < parts.size(); ++j) {
                int id = s.tokens[j];
                if (ID2STR.find(id) == ID2STR.end()) {
                    ID2STR[id] = parts[j];
                }
            }
        }
    }

    std::cout
        << "\nDataset size: "
        << dataset.size()
        << " sentences\n";

    std::cout
        << "\n";

    for (size_t i = 0;
         i < dataset.size();
         i++) {

        std::cout
            << i + 1
            << ". "
            << dataset[i].text
            << "\n";
    }

    LanguageModel model(
        VOCAB_SIZE,
        EMBED_DIM,
        NUM_BLOCKS,
        NUM_HEADS,
        HIDDEN_DIM
    );

    model.SetUseKVCache(false);
    float initial_loss =
        DatasetLoss(
            model,
            dataset
        );

    std::cout
        << "\n============================================================\n"
        << "                 INITIAL MODEL\n"
        << "============================================================\n";

    std::cout
        << "Initial loss: "
        << std::fixed
        << std::setprecision(5)
        << initial_loss
        << "\n";

    std::cout
        << "\n============================================================\n"
        << "                    TRAINING\n"
        << "============================================================\n";

    for (int epoch = 0;
         epoch < EPOCHS;
         epoch++) {

        float epoch_loss = 0.0f;

        for (const Sentence& sentence :
             dataset) {

            epoch_loss +=
                TrainSentence(
                    model,
                    sentence.tokens
                );
        }

        epoch_loss /=
            static_cast<float>(
                dataset.size()
            );

        if (epoch % 100 == 0 ||
            epoch == EPOCHS - 1) {

            float accuracy =
                CalculateAccuracy(
                    model,
                    dataset
                );

            std::cout
                << "Epoch "
                << std::setw(4)
                << epoch
                << " | Loss: "
                << std::fixed
                << std::setprecision(5)
                << epoch_loss
                << " | Accuracy: "
                << std::setprecision(2)
                << accuracy * 100.0f
                << "%\n";
        }
    }

    float final_loss =
        DatasetLoss(
            model,
            dataset
        );

    std::cout
        << "\n============================================================\n"
        << "                    FINAL MODEL\n"
        << "============================================================\n";

    std::cout
        << "Initial loss: "
        << std::fixed
        << std::setprecision(5)
        << initial_loss
        << "\n";

    std::cout
        << "Final loss:   "
        << final_loss
        << "\n";

    float accuracy =
        CalculateAccuracy(
            model,
            dataset
        );

    std::cout
        << "Training accuracy: "
        << std::setprecision(2)
        << accuracy * 100.0f
        << "%\n";

    PrintPredictions(
        model,
        dataset
    );

    TestGeneration(
        model,
        dataset
    );

    bool kv_passed =
        TestKVGeneration(
            model,
            dataset[0]
        );

    bool save_load_passed =
        TestSaveLoad(
            model,
            dataset[0]
        );

    bool training_passed =
        accuracy >=
        TRAIN_ACCURACY_THRESHOLD;

    std::cout
        << "\n============================================================\n"
        << "                    FINAL RESULT\n"
        << "============================================================\n";

    std::cout
        << "Training:       "
        << (training_passed
            ? "PASS"
            : "FAIL")
        << "\n";

    std::cout
        << "KV-cache:       "
        << (kv_passed
            ? "PASS"
            : "FAIL")
        << "\n";

    std::cout
        << "Save / Load:    "
        << (save_load_passed
            ? "PASS"
            : "FAIL")
        << "\n";

    std::cout
        << "\n";

    if (training_passed &&
        kv_passed &&
        save_load_passed) {

        std::cout
            << "============================================================\n"
            << "       LANGUAGE MODEL PASSED FULL TEST\n"
            << "============================================================\n";

    } else {

        std::cout
            << "============================================================\n"
            << "       LANGUAGE MODEL FAILED FULL TEST\n"
            << "============================================================\n";
    }

    return (
        training_passed &&
        kv_passed &&
        save_load_passed
    )
        ? 0
        : 1;
}