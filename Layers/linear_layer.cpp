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
        float max_grad = 0.0f;

        for (size_t i = 0; i < W_->Grad()->GetSize(); i++) {
            float value = W_->Grad()->at(i);

            sum += std::abs(value);
            max_grad = std::max(max_grad, std::abs(value));
        }

        *W_ = *W_ - (*W_->Grad()) * lr;
    }

    if (b_->Grad() != nullptr) {
        float sum = 0.0f;
        float max_grad = 0.0f;

        for (size_t i = 0; i < b_->Grad()->GetSize(); i++) {
            float value = b_->Grad()->at(i);

            sum += std::abs(value);
            max_grad = std::max(max_grad, std::abs(value));
        }

        *b_ = *b_ - (*b_->Grad()) * lr;
    }
}

void LinearLayer::ScaleGrad(float factor) {
    if (W_->Grad() != nullptr) {
        for (size_t i = 0; i < W_->Grad()->GetSize(); i++) {
            W_->Grad()->at(i) *= factor;
        }
    }

    if (b_->Grad() != nullptr) {
        for (size_t i = 0; i < b_->Grad()->GetSize(); i++) {
            b_->Grad()->at(i) *= factor;
        }
    }
}

std::shared_ptr<Tensor> LinearLayer::forward(const Tensor& x) {
    auto x_ptr = std::make_shared<Tensor>(x);
    saved_mult_ = mul_op_.forward({x_ptr, W_});
    saved_added_ = add_op_.forward({saved_mult_, b_});
    return saved_added_;
}

Tensor LinearLayer::backward(const Tensor& grad_output) {
    Tensor grad_x;

    if (saved_added_->GradFn() != nullptr) {
        grad_x = saved_added_->GradFn()->backward(grad_output);
    } else {
        std::vector<size_t> input_shape = saved_added_->GetShape();
        input_shape.back() = input_size_;
        grad_x = Tensor(input_shape, 0.0f);
    }

    float sum = 0.0f;

    for (size_t i = 0; i < grad_x.GetSize(); i++) {
        sum += std::abs(grad_x.at(i));
    }
    return grad_x;
}

void LinearLayer::Save(const std::string& folder, const std::string& name) const {
    W_->SaveTensor(folder + "/" + name + "_W");
    b_->SaveTensor(folder + "/" + name + "_b");
}

void LinearLayer::Load(const std::string& folder, const std::string& name) {
    *W_ = Tensor::LoadTensor(folder + "/" + name + "_W");
    *b_ = Tensor::LoadTensor(folder + "/" + name + "_b");
}

const Tensor& LinearLayer::GetWeights() const { 
    return *W_;
}
const Tensor& LinearLayer::GetBias() const { 
    return *b_;
}