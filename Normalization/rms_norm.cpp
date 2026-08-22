#include "rms_norm.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <cmath>

RMSNorm::RMSNorm(size_t embed_dim)  {
    gamma_ = Tensor({embed_dim}, 1.0f);
}

Tensor RMSNorm::forward(const Tensor& x) {
    std::vector<size_t> shape = x.GetShape();
    size_t batch = shape[0];
    size_t seq_len = shape[1];
    size_t embed_dim = shape[2];

    Tensor result = x;
    saved_rms_ = Tensor({batch, seq_len, 1});

    for (size_t b = 0; b < batch; b++) {
        for (size_t pos = 0; pos < seq_len; pos++) {
            float rms = 0.0f;
            for (size_t i = 0; i < embed_dim; i++) {
                float val = x.at({b, pos, i});
                rms += val * val;
            }
            rms /= embed_dim;
            rms = std::sqrt(rms + 1e-6f);
            saved_rms_.at({b, pos, 0}) = rms;
            for (size_t i = 0; i < embed_dim; i++) {
                result.at({b, pos, i}) /= rms;
                result.at({b, pos, i}) *= gamma_.at(i);
            }
        }
    }
    saved_x_norm_ = result;
    return result;
}

Tensor RMSNorm::backward(const Tensor& grad_output) {
    Tensor grad_gamma = grad_output * saved_x_norm_;
    while (grad_gamma.GetRank() > 1) {
        grad_gamma = grad_gamma.SumAxis(0);
    }

    gamma_.AddGrad(grad_gamma);
    Tensor grad_x_norm = grad_output * gamma_;
    Tensor mean = grad_x_norm * saved_x_norm_;
    mean = mean.Mean(mean.GetShape().size() - 1);
    Tensor grad_x = (1.0f / saved_rms_) * (grad_x_norm - saved_x_norm_ * mean);
    return grad_x;
}

Tensor RMSNorm::GetGamma() {
    return gamma_;
}

void RMSNorm::Update(float lr) {
    if (gamma_.Grad() != nullptr) {
        gamma_ = gamma_ - (*gamma_.Grad()) * lr;
    }
}

void RMSNorm::ClearGrad() {
    gamma_.ClearGrad();
}

void RMSNorm::ScaleGrad(float factor) {
    if (gamma_.Grad() == nullptr) return;
    for (size_t i = 0; i < gamma_.Grad()->GetSize(); i++) {
        gamma_.Grad()->at(i) *= factor;
    }
}

void RMSNorm::Save(const std::string& path) const {
    gamma_.SaveTensor(path);
}

void RMSNorm::Load(const std::string& path) {
    gamma_ = Tensor::LoadTensor(path);
}