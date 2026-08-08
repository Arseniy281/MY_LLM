
#pragma once
#include <vector>

class Tensor;

class Operation {
public:
    virtual Tensor forward(const std::vector<Tensor*>& inputs) = 0;
    virtual void backward(const Tensor& grad_output) const = 0;
    virtual ~Operation() = default;
};