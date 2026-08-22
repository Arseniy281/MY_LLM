#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <random>
#include <string>

#include "../Tensor/tensor.h"
#include "../Attention/rope.h"
#include "../Attention/softmax.h"
#include "../Layers/linear_layer.h"
#include "../Layers/language_model.h"


// ============================================================
// Helpers
// ============================================================

static constexpr float TEST_EPS = 1e-4f;

struct TestResult {
    std::string name;
    bool passed;
};

std::vector<TestResult> results;


bool AlmostEqual(float a, float b, float eps = TEST_EPS) {
    return std::fabs(a - b) <= eps;
}


float MaxDiff(const Tensor& a, const Tensor& b) {
    if (a.GetShape() != b.GetShape()) {
        return INFINITY;
    }

    float max_diff = 0.0f;

    for (size_t i = 0; i < a.GetSize(); i++) {
        float diff = std::fabs(a.at(i) - b.at(i));

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}


float MeanDiff(const Tensor& a, const Tensor& b) {
    if (a.GetShape() != b.GetShape()) {
        return INFINITY;
    }

    if (a.GetSize() == 0) {
        return 0.0f;
    }

    float sum = 0.0f;

    for (size_t i = 0; i < a.GetSize(); i++) {
        sum += std::fabs(a.at(i) - b.at(i));
    }

    return sum / static_cast<float>(a.GetSize());
}


void Report(const std::string& name, bool passed) {
    results.push_back({name, passed});

    std::cout
        << (passed ? "[PASS] " : "[FAIL] ")
        << name
        << "\n";
}


void PrintSeparator() {
    std::cout
        << "------------------------------------------------------------\n";
}


// ============================================================
// TEST 1: Tensor
// ============================================================

bool TestTensorBasic() {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "TEST 1: Tensor basic operations\n";
    PrintSeparator();

    Tensor a({2, 3});

    for (size_t i = 0; i < a.GetSize(); i++) {
        a.at(i) = static_cast<float>(i + 1);
    }

    bool shape_ok =
        a.GetRank() == 2 &&
        a.GetShape()[0] == 2 &&
        a.GetShape()[1] == 3 &&
        a.GetSize() == 6;

    Tensor b({2, 3}, 2.0f);

    Tensor c = a + b;

    bool values_ok =
        AlmostEqual(c.at({0, 0}), 3.0f) &&
        AlmostEqual(c.at({0, 1}), 4.0f) &&
        AlmostEqual(c.at({1, 2}), 8.0f);

    Tensor d = a * 2.0f;

    bool mul_ok =
        AlmostEqual(d.at({0, 0}), 2.0f) &&
        AlmostEqual(d.at({1, 2}), 12.0f);

    Tensor e = a / 2.0f;

    bool div_ok =
        AlmostEqual(e.at({0, 0}), 0.5f) &&
        AlmostEqual(e.at({1, 2}), 3.0f);

    Tensor t = a.Transpose();

    bool transpose_ok =
        t.GetShape()[0] == 3 &&
        t.GetShape()[1] == 2 &&
        AlmostEqual(t.at({0, 0}), 1.0f) &&
        AlmostEqual(t.at({0, 1}), 4.0f) &&
        AlmostEqual(t.at({2, 1}), 6.0f);

    bool passed =
        shape_ok &&
        values_ok &&
        mul_ok &&
        div_ok &&
        transpose_ok;

    std::cout << "shape:      " << (shape_ok ? "OK" : "FAIL") << "\n";
    std::cout << "addition:   " << (values_ok ? "OK" : "FAIL") << "\n";
    std::cout << "multiply:   " << (mul_ok ? "OK" : "FAIL") << "\n";
    std::cout << "divide:     " << (div_ok ? "OK" : "FAIL") << "\n";
    std::cout << "transpose:  " << (transpose_ok ? "OK" : "FAIL") << "\n";

    Report("Tensor basic operations", passed);

    return passed;
}


// ============================================================
// TEST 2: RoPE
// ============================================================

bool TestRoPE() {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "TEST 2: RoPE\n";
    PrintSeparator();

    // batch = 2
    // seq_len = 4
    // head_dim = 8
    Tensor x({2, 4, 8});

    for (size_t i = 0; i < x.GetSize(); i++) {
        x.at(i) =
            std::sin(static_cast<float>(i) * 0.37f) +
            0.1f * static_cast<float>(i);
    }

    // --------------------------------------------------------
    // 2.1 RoPE should preserve vector norm for every pair
    // --------------------------------------------------------

    Tensor rotated = RoPE(x, 0);

    float max_norm_diff = 0.0f;

    for (size_t b = 0; b < 2; b++) {
        for (size_t s = 0; s < 4; s++) {
            for (size_t d = 0; d < 8; d += 2) {

                float x1 = x.at({b, s, d});
                float x2 = x.at({b, s, d + 1});

                float y1 = rotated.at({b, s, d});
                float y2 = rotated.at({b, s, d + 1});

                float before = x1 * x1 + x2 * x2;
                float after = y1 * y1 + y2 * y2;

                float diff = std::fabs(before - after);

                max_norm_diff =
                    std::max(max_norm_diff, diff);
            }
        }
    }

    bool norm_ok = max_norm_diff < 1e-4f;

    std::cout
        << "Norm preservation max diff: "
        << max_norm_diff
        << "\n";

    // --------------------------------------------------------
    // 2.2 RoPE backward should undo RoPE
    // --------------------------------------------------------

    Tensor restored = RoPE_backward(rotated, 0);

    float restore_diff = MaxDiff(x, restored);

    bool inverse_ok = restore_diff < 1e-4f;

    std::cout
        << "RoPE -> backward -> original max diff: "
        << restore_diff
        << "\n";

    // --------------------------------------------------------
    // 2.3 start_pos should actually change rotation
    // --------------------------------------------------------

    Tensor rope0 = RoPE(x, 0);
    Tensor rope10 = RoPE(x, 10);

    float position_diff = MaxDiff(rope0, rope10);

    bool position_ok = position_diff > 1e-5f;

    std::cout
        << "start_pos=0 vs start_pos=10 max diff: "
        << position_diff
        << "\n";

    // --------------------------------------------------------
    // 2.4 start_pos + backward
    // --------------------------------------------------------

    Tensor rope100 = RoPE(x, 100);
    Tensor restored100 = RoPE_backward(rope100, 100);

    float restore100_diff = MaxDiff(x, restored100);

    bool inverse100_ok = restore100_diff < 1e-4f;

    std::cout
        << "RoPE(start=100) inverse max diff: "
        << restore100_diff
        << "\n";

    bool passed =
        norm_ok &&
        inverse_ok &&
        position_ok &&
        inverse100_ok;

    Report("RoPE correctness", passed);

    return passed;
}


// ============================================================
// TEST 3: Softmax
// ============================================================

bool TestSoftmax() {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "TEST 3: Softmax\n";
    PrintSeparator();

    Softmax softmax;

    Tensor x({5});

    x.at(0) = 1.0f;
    x.at(1) = 2.0f;
    x.at(2) = 3.0f;
    x.at(3) = -2.0f;
    x.at(4) = 0.5f;

    Tensor y = softmax.forward(x);

    float sum = 0.0f;
    float min_value = y.at(0);
    float max_value = y.at(0);

    for (size_t i = 0; i < y.GetSize(); i++) {
        sum += y.at(i);

        min_value =
            std::min(min_value, y.at(i));

        max_value =
            std::max(max_value, y.at(i));
    }

    bool sum_ok = std::fabs(sum - 1.0f) < 1e-5f;
    bool range_ok =
        min_value >= 0.0f &&
        max_value <= 1.0f;

    std::cout << "Probabilities: ";

    for (size_t i = 0; i < y.GetSize(); i++) {
        std::cout << y.at(i) << " ";
    }

    std::cout << "\n";

    std::cout
        << "Sum = "
        << sum
        << "\n";

    // --------------------------------------------------------
    // Softmax backward basic sanity check
    //
    // For grad_output = [1, 1, 1, 1, 1],
    // derivative of sum(softmax(x)) is zero.
    // --------------------------------------------------------

    Tensor grad({5}, 1.0f);

    Tensor grad_x = softmax.backward(grad);

    float max_grad = 0.0f;

    for (size_t i = 0; i < grad_x.GetSize(); i++) {
        max_grad =
            std::max(max_grad, std::fabs(grad_x.at(i)));
    }

    bool backward_ok = max_grad < 1e-5f;

    std::cout
        << "Softmax backward max diff from zero: "
        << max_grad
        << "\n";

    bool passed =
        sum_ok &&
        range_ok &&
        backward_ok;

    Report("Softmax correctness", passed);

    return passed;
}


// ============================================================
// TEST 4: Top-P
// ============================================================

bool TestTopP() {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "TEST 4: Top-P\n";
    PrintSeparator();

    // Small model just to access public TopP().
    LanguageModel model(
        6,      // vocab
        8,      // embed dim
        1,      // blocks
        1,      // heads
        16      // hidden
    );

    Tensor probs({6});

    probs.at(0) = 0.40f;
    probs.at(1) = 0.25f;
    probs.at(2) = 0.15f;
    probs.at(3) = 0.10f;
    probs.at(4) = 0.06f;
    probs.at(5) = 0.04f;

    model.TopP(probs, 0.81f);

    float sum = 0.0f;
    size_t non_zero = 0;

    for (size_t i = 0; i < probs.GetSize(); i++) {
        sum += probs.at(i);

        if (probs.at(i) > 0.0f) {
            non_zero++;
        }
    }

    std::cout << "After Top-P:\n";

    for (size_t i = 0; i < probs.GetSize(); i++) {
        std::cout
            << "  token "
            << i
            << ": "
            << probs.at(i)
            << "\n";
    }

    std::cout << "Sum: " << sum << "\n";
    std::cout << "Non-zero tokens: " << non_zero << "\n";

    bool sum_ok =
        std::fabs(sum - 1.0f) < 1e-5f;

    // 0.40 + 0.25 + 0.15 = 0.80
    // Therefore three tokens should survive.
    bool border_ok = non_zero == 4;

    bool tail_zero =
        probs.at(4) == 0.0f &&
        probs.at(5) == 0.0f;

    bool passed =
        sum_ok &&
        border_ok &&
        tail_zero;

    Report("Top-P correctness", passed);

    return passed;
}


// ============================================================
// Helper for creating token tensor
// ============================================================

Tensor MakeTokenTensor(
    const std::vector<int>& tokens,
    size_t begin,
    size_t end
) {
    size_t len = end - begin;

    Tensor result({1, len});

    for (size_t i = 0; i < len; i++) {
        result.at({0, i}) =
            static_cast<float>(tokens[begin + i]);
    }

    return result;
}


// ============================================================
// TEST 5: FULL FORWARD vs KV-CACHE
//
// This is the important one.
//
// We compare the actual logits, not only argmax.
// ============================================================

bool TestFullVsCache(
    LanguageModel& model,
    const std::vector<int>& tokens
) {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "TEST 5: FULL FORWARD vs KV-CACHE\n";
    PrintSeparator();

    if (tokens.empty()) {
        std::cout << "Token sequence is empty.\n";
        Report("FULL vs KV-CACHE", false);
        return false;
    }

    const float tolerance = 1e-5f;

    // --------------------------------------------------------
    // FULL FORWARD
    // --------------------------------------------------------

    std::vector<Tensor> full_logits;

    for (size_t pos = 0; pos < tokens.size(); pos++) {

        model.ResetCache();

        Tensor input =
            MakeTokenTensor(tokens, 0, pos + 1);

        auto output = model.forward(input);

        size_t last_pos =
            output->GetShape()[1] - 1;

        Tensor logits({output->GetShape()[2]});

        for (size_t v = 0; v < output->GetShape()[2]; v++) {
            logits.at(v) =
                output->at({0, last_pos, v});
        }

        full_logits.push_back(logits);
    }

    // --------------------------------------------------------
    // KV CACHE FORWARD
    // --------------------------------------------------------

    model.ResetCache();

    std::vector<Tensor> cache_logits;

    for (size_t pos = 0; pos < tokens.size(); pos++) {

        Tensor input({1, 1});

        input.at({0, 0}) =
            static_cast<float>(tokens[pos]);

        auto output = model.forward(input);

        size_t last_pos =
            output->GetShape()[1] - 1;

        Tensor logits({output->GetShape()[2]});

        for (size_t v = 0; v < output->GetShape()[2]; v++) {
            logits.at(v) =
                output->at({0, last_pos, v});
        }

        cache_logits.push_back(logits);
    }

    // --------------------------------------------------------
    // Compare
    // --------------------------------------------------------

    bool passed = true;

    float global_max_diff = 0.0f;
    float global_mean_diff = 0.0f;

    size_t worst_position = 0;

    for (size_t pos = 0; pos < tokens.size(); pos++) {

        float max_diff =
            MaxDiff(full_logits[pos], cache_logits[pos]);

        float mean_diff =
            MeanDiff(full_logits[pos], cache_logits[pos]);

        global_mean_diff += mean_diff;

        if (max_diff > global_max_diff) {
            global_max_diff = max_diff;
            worst_position = pos;
        }

        bool ok = max_diff <= tolerance;

        if (!ok) {
            passed = false;
        }

        std::cout
            << "Position "
            << pos
            << " | input="
            << tokens[pos]
            << " | max_diff="
            << std::scientific
            << max_diff
            << " | mean_diff="
            << mean_diff
            << " | "
            << (ok ? "OK" : "FAIL")
            << std::defaultfloat
            << "\n";
    }

    global_mean_diff /= static_cast<float>(tokens.size());

    std::cout << "\n";
    std::cout
        << "Global max diff: "
        << std::scientific
        << global_max_diff
        << "\n";

    std::cout
        << "Global mean diff: "
        << global_mean_diff
        << "\n";

    std::cout
        << "Worst position: "
        << worst_position
        << "\n";

    std::cout << std::defaultfloat;

    Report("FULL vs KV-CACHE", passed);

    return passed;
}


// ============================================================
// TEST 6: Several sequence lengths
// ============================================================

bool TestDifferentSequenceLengths(
    LanguageModel& model,
    const std::vector<int>& tokens
) {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "TEST 6: Different sequence lengths\n";
    PrintSeparator();

    std::vector<size_t> lengths = {
        1,
        2,
        3,
        5,
        8,
        10,
        15,
        20
    };

    bool all_passed = true;

    for (size_t len : lengths) {

        if (len > tokens.size()) {
            continue;
        }

        std::vector<int> current(
            tokens.begin(),
            tokens.begin() + len
        );

        std::cout
            << "\nSequence length = "
            << len
            << "\n";

        bool passed =
            TestFullVsCache(model, current);

        if (!passed) {
            all_passed = false;
        }
    }

    Report(
        "FULL vs CACHE on different sequence lengths",
        all_passed
    );

    return all_passed;
}


// ============================================================
// TEST 7: Generation sanity
// ============================================================

bool TestGeneration(
    LanguageModel& model,
    int start_token
) {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "TEST 7: Generation sanity\n";
    PrintSeparator();

    model.ResetCache();

    std::vector<int> generated =
        model.generate(
            start_token,
            30,
            1.0f,
            0.0f,
            -1
        );

    bool non_empty =
        !generated.empty();

    bool correct_start =
        non_empty &&
        generated[0] == start_token;

    bool correct_length =
        generated.size() <= 30;

    std::cout
        << "Generated size: "
        << generated.size()
        << "\n";

    std::cout << "Tokens: ";

    for (int token : generated) {
        std::cout << token << " ";
    }

    std::cout << "\n";

    bool passed =
        non_empty &&
        correct_start &&
        correct_length;

    Report("Generation sanity", passed);

    return passed;
}


// ============================================================
// TEST 8: Sampling sanity
// ============================================================

bool TestSampling() {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "TEST 8: Sampling\n";
    PrintSeparator();

    LanguageModel model(
        4,
        8,
        1,
        1,
        16
    );

    Tensor probs({4});

    probs.at(0) = 0.05f;
    probs.at(1) = 0.10f;
    probs.at(2) = 0.25f;
    probs.at(3) = 0.60f;

    // Greedy must always select token 3.
    bool greedy_ok = true;

    for (int i = 0; i < 100; i++) {
        int token =
            model.SampleGreedy(probs);

        if (token != 3) {
            greedy_ok = false;
            break;
        }
    }

    std::cout
        << "Greedy always selects max probability: "
        << (greedy_ok ? "YES" : "NO")
        << "\n";

    // Sampling should produce more than one token
    // with this distribution.
    bool sampled_multiple = false;

    int first = model.Sample(probs);

    for (int i = 0; i < 200; i++) {
        int token =
            model.Sample(probs);

        if (token != first) {
            sampled_multiple = true;
            break;
        }
    }

    std::cout
        << "Sampling produces different tokens: "
        << (sampled_multiple ? "YES" : "NO")
        << "\n";

    bool passed =
        greedy_ok &&
        sampled_multiple;

    Report("Sampling sanity", passed);

    return passed;
}


// ============================================================
// MAIN
// ============================================================

int main() {

    std::cout
        << "\n"
        << "============================================================\n"
        << "              TRANSFORMER CORRECTNESS TESTS\n"
        << "============================================================\n";

    // --------------------------------------------------------
    // Basic tests that don't require a trained model
    // --------------------------------------------------------

    TestTensorBasic();

    TestRoPE();

    TestSoftmax();

    TestTopP();

    TestSampling();

    // --------------------------------------------------------
    // Create model
    // --------------------------------------------------------

    std::cout << "\n";
    PrintSeparator();
    std::cout << "Creating LanguageModel for integration tests\n";
    PrintSeparator();

    const size_t vocab_size = 77;
    const size_t embed_dim = 32;
    const size_t num_blocks = 2;
    const size_t num_heads = 4;
    const size_t hidden_dim = 64;

    LanguageModel model(
        vocab_size,
        embed_dim,
        num_blocks,
        num_heads,
        hidden_dim
    );

    /*
        IMPORTANT:

        These dimensions are only used for the correctness test.

        If your actual test model uses different dimensions,
        change the five constants above to the same values.
    */

    std::vector<int> tokens = {
        7, 4, 11, 11, 14,
        22, 14, 17, 11, 3,
        14, 22, 14, 17, 11,
        3, 14, 22, 14, 17
    };

    std::cout << "Test tokens: ";

    for (int token : tokens) {
        std::cout << token << " ";
    }

    std::cout << "\n";

    // --------------------------------------------------------
    // FULL vs CACHE
    // --------------------------------------------------------

    TestFullVsCache(model, tokens);

    // --------------------------------------------------------
    // Different lengths
    // --------------------------------------------------------

    TestDifferentSequenceLengths(model, tokens);

    // --------------------------------------------------------
    // Generation
    // --------------------------------------------------------

    TestGeneration(model, tokens[0]);

    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------

    std::cout << "\n";
    std::cout
        << "============================================================\n";
    std::cout
        << "                         SUMMARY\n";
    std::cout
        << "============================================================\n";

    size_t passed = 0;
    size_t failed = 0;

    for (const auto& result : results) {

        if (result.passed) {
            passed++;
        } else {
            failed++;
        }

        std::cout
            << (result.passed ? "[PASS] " : "[FAIL] ")
            << result.name
            << "\n";
    }

    std::cout
        << "\nPassed: "
        << passed
        << "\n";

    std::cout
        << "Failed: "
        << failed
        << "\n";

    if (failed == 0) {
        std::cout
            << "\nALL TESTS PASSED.\n";
    } else {
        std::cout
            << "\nSOME TESTS FAILED.\n";
    }

    std::cout
        << "============================================================\n";

    return failed == 0 ? 0 : 1;
}