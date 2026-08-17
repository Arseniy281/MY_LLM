#pragma once
#include "../Autograd/add_op.h"
#include "../Autograd/mul_op.h"
#include "../Tensor/tensor.h"
#include <memory>

class LinearLayer {
private:
    std::shared_ptr<Tensor> W_;
    std::shared_ptr<Tensor> b_;
    size_t input_size_;
    size_t  output_size_;
    MulOp mul_op_;
    AddOp add_op_;

    std::shared_ptr<Tensor> saved_mult_;
    std::shared_ptr<Tensor> saved_added_;
public:
    LinearLayer(size_t in, size_t out);
    LinearLayer() = default;

    void ClearGrad();
    void Update(float lr);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);
    Tensor backward(const Tensor& grad_output);
};