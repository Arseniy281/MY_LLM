#include "feed_forward.h"
#include <memory>
#include <string>
#include <sys/stat.h>
#include <errno.h> 


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

void FeedForward::ScaleGrad(float factor) {
    fc1_.ScaleGrad(factor);
    fc2_.ScaleGrad(factor);
}

void FeedForward::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + folder);
    }

    fc1_.Save(folder, "fc1");
    fc2_.Save(folder, "fc2");
}

void FeedForward::Load(const std::string& folder) {
    fc1_.Load(folder, "fc1");
    fc2_.Load(folder, "fc2");
}