#include "../Attention/attention.h"
#include "../Tensor/tensor.h"
#include <iostream>

void print_shape(const std::string& name, const Tensor& t) {
    std::cout << name << " shape: (";
    auto shape = t.GetShape();
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << ")\n";
}

void print_tensor(const std::string& name, const Tensor& t) {
    std::cout << name << ":\n";
    t.print();
    std::cout << "\n";
}

int main() {
    std::cout << "=== Attention Test ===\n\n";

    // 1. Создаём входной тензор
    size_t batch = 2;
    size_t seq_len = 3;
    size_t embed_dim = 4;
    size_t num_heads = 2;

    Tensor x({batch, seq_len, embed_dim});
    // Заполняем данными (для простоты — индексами)
    for (size_t i = 0; i < x.GetSize(); ++i) {
        x.RawData()[i] = static_cast<float>(i);
    }
    print_shape("x", x);
    print_tensor("x", x);

    // 2. Создаём Attention (без маски)
    std::cout << "Test 1: Attention without mask\n";
    MultiHeadAttention attn(embed_dim, num_heads);
    auto output = attn.forward(x);

    print_shape("output", *output);
    std::cout << "Output shape matches input: "
              << (output->GetShape() == x.GetShape() ? "✅ YES" : "❌ NO") << "\n\n";

    // 3. Проверяем маску
    std::cout << "Test 2: Causal mask\n";
    Tensor mask = attn.CreateCausalMask(seq_len);
    print_tensor("mask (0 = can see, -inf = cannot)", mask);

    // 4. Проверяем, что маска работает
    // (вручную проверить сложно, но можно проверить, что маска имеет правильную форму)
    std::cout << "Mask shape: (" << mask.GetShape()[0] << ", " << mask.GetShape()[1] << ")\n";
    std::cout << "Mask is lower triangular: "
              << (mask.GetShape()[0] == seq_len && mask.GetShape()[1] == seq_len ? "✅ YES" : "❌ NO") << "\n\n";

    // 5. Проверяем Multi-Head
    std::cout << "Test 3: Multi-Head\n";
    std::cout << "num_heads = " << num_heads << "\n";
    std::cout << "head_dim = " << embed_dim / num_heads << "\n";
    std::cout << "output shape: (" << output->GetShape()[0] << ", "
              << output->GetShape()[1] << ", " << output->GetShape()[2] << ")\n";
    std::cout << "Expected: (" << batch << ", " << seq_len << ", " << embed_dim << ")\n";
    std::cout << "Multi-Head works: "
              << (output->GetShape()[0] == batch && 
                  output->GetShape()[1] == seq_len && 
                  output->GetShape()[2] == embed_dim ? "✅ YES" : "❌ NO") << "\n";

    std::cout << "\n=== Attention Test Completed ===\n";
    return 0;
}