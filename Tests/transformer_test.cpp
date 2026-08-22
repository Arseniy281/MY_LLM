#include "../Transformer/transformer.h"
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

int main() {
    std::cout << "=== Transformer Test ===\n\n";

    size_t num_blocks = 2;
    size_t batch = 2;
    size_t seq_len = 4;
    size_t embed_dim = 8;
    size_t num_heads = 2;
    size_t hidden_dim = 32;

    Tensor x({batch, seq_len, embed_dim});
    for (size_t i = 0; i < x.GetSize(); ++i) {
        x.RawData()[i] = static_cast<float>(i) / 10.0f;
    }
    print_shape("x", x);

    Transformer model(num_blocks, embed_dim, num_heads, hidden_dim);

    auto y = model.forward(x);
    print_shape("y", *y);

    if (y->GetShape() == x.GetShape()) {
        std::cout << "✅ Shape preserved through " << num_blocks << " blocks\n";
    }

    Tensor grad_output({batch, seq_len, embed_dim}, 1.0f);
    Tensor grad_x = model.backward(grad_output);
    print_shape("grad_x", grad_x);

    if (grad_x.GetShape() == x.GetShape()) {
        std::cout << "✅ grad_x shape matches input\n";
    }

    model.Update(0.01f);
    model.ClearGrad();
    std::cout << "✅ Update + ClearGrad passed\n";

    std::cout << "\n=== Transformer Test Completed ===\n";
    return 0;
}