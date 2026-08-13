#include<../Tensor/tensor.h>

class PositionalEncoding {
public:
    Tensor forward(size_t seq_len, size_t embed_dim);
};