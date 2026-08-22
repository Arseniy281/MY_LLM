#include "../Layers/linear_layer.h"
#include "../Autograd/gelu.h"
#include "../Tensor/tensor.h"
#include <string>

class FeedForward {
private:
    LinearLayer fc1_;
    LinearLayer fc2_;
    Gelu gelu_;
public:
    FeedForward(size_t embed_dim, size_t hidden_dim);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);
    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& folder) const;
    void Load(const std::string& folder);
};