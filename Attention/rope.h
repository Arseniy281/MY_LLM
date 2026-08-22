#include "../Tensor/tensor.h"

Tensor RoPE(const Tensor& tensor, size_t start_pos);
Tensor RoPE_backward(const Tensor& tensor, size_t start_pos);