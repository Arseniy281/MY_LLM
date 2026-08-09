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

std::shared_ptr<Tensor> LinearLayer::forward(const std::shared_ptr<Tensor>& x) {
    auto mult = mul_op_.forward({x, W_});
    auto added = add_op_.forward({mult, b_});
        
    return added;
}