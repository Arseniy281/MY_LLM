#include "../Autograd/add_op.h"
#include "../Autograd/mul_op.h"
#include "../Tensor/tensor.h"
#include "linear_layer.h"
#include <memory>

LinearLayer::LinearLayer(size_t in, size_t out) : input_size_(in), output_size_(out) {
    W_ = std::make_shared<Tensor>(Tensor::Random({in, out}, -0.5f, 0.5f));
    b_ = std::make_shared<Tensor>(Tensor({1, out}, 0.0f)); 
}

void LinearLayer::ClearGrad() {
    W_->ClearGrad();
    b_->ClearGrad();
}

void LinearLayer::Update(float lr) {
    if (W_->Grad() != nullptr) {
        float sum = 0.0f;
        for (size_t i = 0; i < W_->Grad()->GetSize(); ++i) {
            sum += W_->Grad()->at(i);
        }        
        *W_ = *W_ - (*W_->Grad()) * lr;
    }
    if (b_->Grad() != nullptr) {
        *b_ = *b_ - (*b_->Grad()) * lr;
    }
}

std::shared_ptr<Tensor> LinearLayer::forward(const Tensor& x) {
    auto x_ptr = std::make_shared<Tensor>(x);
    saved_mult_ = mul_op_.forward({x_ptr, W_});
    saved_added_ = add_op_.forward({saved_mult_, b_});
    return saved_added_;
}

Tensor LinearLayer::backward(const Tensor& grad_output) {
    if (saved_added_->GradFn() != nullptr) {
        saved_added_->GradFn()->backward(grad_output);
    }

    Tensor grad_x;
    if (saved_mult_->GradFn() != nullptr && saved_mult_->Grad() != nullptr) {
        grad_x = saved_mult_->GradFn()->backward(*saved_mult_->Grad());
    } else {
        std::vector<size_t> input_shape = saved_mult_->GetShape();
        input_shape.back() = input_size_;
        grad_x = Tensor(input_shape, 0.0f);
    }

    return grad_x;
}