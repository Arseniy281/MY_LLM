#pragma once
#include "operation.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>


class SumOp : Operation {
private:
    std::shared_ptr<Tensor> parent_;
public:
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    Tensor backward(const Tensor& grad_output) override;
};