#include "feed_forward.h"
#include <memory>

FeedForward::FeedForward(size_t embed_dim, size_t hidden_dim) {
    fc1_ = LinearLayer(embed_dim, hidden_dim);
    fc2_ = LinearLayer(hidden_dim, embed_dim);
}

Tensor FeedForward::forward(const Tensor& x) {
    auto hidden = fc1_.forward(x);
    auto activated = gelu_.forward({hidden});
    auto output = fc2_.forward(*activated);
    return *output;
}

Tensor FeedForward::backward(const Tensor& grad_output) {
    Tensor grad_activated = fc2_.backward(grad_output);
    Tensor grad_hidden = gelu_.backward(grad_activated);    
    Tensor grad_x = fc1_.backward(grad_hidden);
    
    return grad_x;
}

void FeedForward::Update(float lr) {
    fc1_.Update(lr);
    fc2_.Update(lr);
}

void FeedForward::ClearGrad() {
    fc1_.ClearGrad();
    fc2_.ClearGrad();
}