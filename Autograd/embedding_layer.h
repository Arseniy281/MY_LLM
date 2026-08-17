#include "../Tensor/tensor.h"
#include "operation.h"


class EmbeddingLayer : Operation {
private:
    Tensor embeddings_;
    size_t vocab_size_;
    size_t embedding_dim_;
    std::shared_ptr<Tensor> indices_;

    std::shared_ptr<Tensor> grad_ = nullptr;
    
public:
    EmbeddingLayer(size_t vocab_size, size_t embedding_dim);

    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& indices) override;
    Tensor backward(const Tensor& grad_output);

    void ClearGrad();
    void Update(float lr);
};