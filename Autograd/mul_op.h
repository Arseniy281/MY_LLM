#pragma once
#include "operation.h"
#include "../Tensor/tensor.h"
#include <vector>

class MulOp : public Operation {
private:
    Tensor* first_;
    Tensor* second_;
public:
    Tensor forward(const std::vector<Tensor*>& inputs) override;
    void backward(const Tensor& grad_output) const override;
};