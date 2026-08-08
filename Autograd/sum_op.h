#pragma once
#include "operation.h"
#include "../Tensor/tensor.h"
#include <vector>


class SumOp : Operation {
private:
    Tensor* parent_;
public:
    Tensor forward(const std::vector<Tensor*>& inputs) override;
    void backward(const Tensor& grad_output) const override;
};