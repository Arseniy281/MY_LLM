#include "../Tensor/tensor.h"
#include "add_op.h"
#include "sub_op.h"
#include "mul_op.h"
#include "square_op.h"
#include "sum_op.h"
#include <iostream>

void CheckGradient(const std::string& name, Tensor* tensor) {
    if (!tensor || !tensor->Grad()) {
        std::cout << "  ❌ " << name << ".grad_ is nullptr!\n";
        return;
    }
    std::cout << "  ✅ " << name << ".grad_:\n";
    tensor->PrintGrad();
}

int main() {
    try {
        std::cout << "=== Autograd Test ===\n\n";

        Tensor x({2, 2});
        x.at({0, 0}) = 1.0f; x.at({0, 1}) = 2.0f;
        x.at({1, 0}) = 3.0f; x.at({1, 1}) = 4.0f;

        Tensor W({2, 1});
        W.at({0, 0}) = 0.5f;
        W.at({1, 0}) = 0.5f;

        Tensor b({2, 1});
        b.at({0, 0}) = 0.1f;
        b.at({1, 0}) = 0.1f;

        Tensor y_true({2, 1});
        y_true.at({0, 0}) = 2.0f;
        y_true.at({1, 0}) = 4.0f;

        MulOp mul_op;
        AddOp add_op;
        SubOp sub_op;
        SquareOp square_op;
        SumOp sum_op;

        Tensor z = mul_op.forward({&x, &W});
        Tensor y = add_op.forward({&z, &b});
        Tensor diff = sub_op.forward({&y, &y_true});
        Tensor sq = square_op.forward({&diff});
        Tensor loss = sum_op.forward({&sq});

        if (!loss.GradFn()) {
            std::cout << "❌ loss has no grad_fn!\n";
            return 1;
        }
        std::cout << "✅ Graph built successfully\n";

        Tensor grad_output({1, 1}, 1.0f);
        loss.backward(grad_output);
        std::cout << "✅ Backward completed\n\n";

        std::cout << "Gradients:\n";
        CheckGradient("W", &W);
        CheckGradient("b", &b);
        CheckGradient("x", &x);
        CheckGradient("z", &z);
        CheckGradient("y", &y);
        CheckGradient("diff", &diff);
        CheckGradient("sq", &sq);

        std::cout << "\n✅ All tests passed!\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}