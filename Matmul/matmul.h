#pragma once
#include "../Tensor/tensor.h"
#include <vector>
#include <thread>

Tensor MatMul(const Tensor& first, const Tensor& second);
Tensor MatMulMultithreaded(const Tensor& first, const Tensor& second);
Tensor MatMulLoopTiling(const Tensor& first, const Tensor& second, size_t block_size = 64);