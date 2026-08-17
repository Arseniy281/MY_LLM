#pragma once
#include <vector>

class Tensor;

class Operation {
public:
    virtual std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual ~Operation() = default;
};