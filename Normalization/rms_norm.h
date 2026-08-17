#include "../Tensor/tensor.h"

class RMSNorm {
private:
    Tensor gamma_;
    Tensor saved_x_norm_;
    Tensor saved_rms_;
public:
    RMSNorm(size_t embed_dim);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    Tensor GetGamma();
};