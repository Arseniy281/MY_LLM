#include "../Tensor/tensor.h"
#include <memory>

class Softmax {
    std::shared_ptr<Tensor> result_;
public:
    Tensor forward(const Tensor& matrix);
    Tensor backward(const Tensor& grad_output);
};