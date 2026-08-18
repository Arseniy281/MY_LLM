#include "transformer_block.h"
#include "../Tensor/tensor.h"
#include <iostream>
#include <cmath>

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
    std::cout << "=== TransformerBlock Test ===\n\n";

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
    std::cout << "x[0,0,0] = " << x.at({0, 0, 0}) << "\n\n";

    TransformerBlock block(embed_dim, num_heads, hidden_dim);

    Tensor y = block.forward(x);
    print_shape("y", y);
    std::cout << "y[0,0,0] = " << y.at({0, 0, 0}) << "\n\n";

    if (y.GetShape() == x.GetShape()) {
        std::cout << "✅ Shape preserved: " << y.GetShape().size() << "D\n";
    } else {
        std::cout << "❌ Shape mismatch!\n";
    }

    Tensor grad_output({batch, seq_len, embed_dim}, 1.0f);
    Tensor grad_x = block.backward(grad_output);
    print_shape("grad_x", grad_x);
    std::cout << "grad_x[0,0,0] = " << grad_x.at({0, 0, 0}) << "\n\n";

    if (grad_x.GetShape() == x.GetShape()) {
        std::cout << "✅ grad_x shape matches input\n";
    } else {
        std::cout << "❌ grad_x shape mismatch!\n";
    }

    block.Update(0.01f);
    block.ClearGrad();
    std::cout << "✅ Update + ClearGrad passed\n";

    std::cout << "\n=== TransformerBlock Test Completed ===\n";
    return 0;
}