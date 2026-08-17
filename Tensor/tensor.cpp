#include "tensor.h"

size_t Tensor::ComputeIndex(const std::vector<size_t>& indexes) const {
    if (indexes.size() != rank_) {
        throw std::runtime_error("Number of indexes must match rank");
    }
    
    size_t index = 0;
    size_t step = 1;
    
    for (int i = rank_ - 1; i >= 0; i--) {
        if (indexes[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds");
        }
        index += indexes[i] * step;
        step *= shape_[i];
    }
    
    return index;
}

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_ = std::vector<float>(size_, 0.0);
    rank_ = shape_.size();
}

Tensor::Tensor(std::vector<size_t> shape, float k) : shape_(std::move(shape)) {
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_ = std::vector<float>(size_, k);
    rank_ = shape_.size();
}

Tensor Tensor::Random(std::vector<size_t> shape, float min, float max) {
    Tensor result(shape);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < result.size_; i++) {
        result.data_[i] = dist(gen);
    }
    return result;
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data)
        : shape_(std::move(shape)), data_(std::move(data)) {

    rank_ = shape_.size();
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    if (data_.size() < size_) {
        data_.resize(size_, 0.0f);
    } else if (data_.size() > size_) {
        data_.resize(size_);
    }
}


float& Tensor::at(size_t index) {
    return data_[index];
}
const float& Tensor::at(size_t index) const {
    return data_[index];
}

float& Tensor::at(const std::vector<size_t>& indexes) {
    return data_[ComputeIndex(indexes)];
}

const float& Tensor::at(const std::vector<size_t>& indexes) const {
    return data_[ComputeIndex(indexes)];
}

void Tensor::Set(std::vector<size_t> indexes, float value) {
    data_[ComputeIndex(indexes)] = value;
}

auto Tensor::GetIter(const std::vector<size_t>& indexes) {
    return data_.begin() + ComputeIndex(indexes);
}

const auto Tensor::GetIter(const std::vector<size_t>& indexes) const {
    return data_.begin() + ComputeIndex(indexes);
}

void Tensor::Reshape(std::vector<size_t> new_shape) {
    size_t first_size = std::accumulate(shape_.begin(), 
        shape_.end(), 1, std::multiplies<size_t>());
    size_t second_size = std::accumulate(new_shape.begin(), 
        new_shape.end(), 1, std::multiplies<size_t>());
    if (first_size != second_size) {
        throw std::runtime_error("This reshape is not possivle");
    }
    shape_ = new_shape;
    rank_ = shape_.size();
}

Tensor Tensor::Reshape(std::vector<size_t> new_shape) const {
    size_t first_size = std::accumulate(shape_.begin(), 
        shape_.end(), 1, std::multiplies<size_t>());
    size_t second_size = std::accumulate(new_shape.begin(), 
        new_shape.end(), 1, std::multiplies<size_t>());
    if (first_size != second_size) {
        throw std::runtime_error("This reshape is not possivle");
    }
    Tensor tensor(*this);
    tensor.shape_ = new_shape;
    tensor.rank_ = tensor.shape_.size();
    return tensor;
}

size_t Tensor::GetSize() const {
    return size_;
}

size_t Tensor::GetRank() const {
    return rank_;
}

const std::vector<size_t>& Tensor::GetShape() const {
    return shape_;
}

std::vector<float> Tensor::Data() {
    return data_;
}

const std::vector<float> Tensor::Data() const {
    return data_;
}

float* Tensor::RawData() {
    return data_.data();
}

const float* Tensor::RawData() const {
    return data_.data();
}

std::shared_ptr<Tensor> Tensor::Grad() const {
    return grad_;
}

void Tensor::ClearGrad() {
    grad_ = nullptr;
}

Operation* Tensor::GradFn() const {
    return grad_fn_;
}

void Tensor::backward(const Tensor& grad_output) {
    if (grad_fn_ == nullptr) {
        throw std::runtime_error("This tensor has no grad_fn");
    }
    grad_fn_->backward(grad_output);
}

Tensor Tensor::SumAxis(int axis) {
    if (axis >= rank_) {
        throw std::runtime_error("Axis is bigger, than rank");
    }
    std::vector<size_t> result_shape = shape_;
    result_shape.erase(result_shape.begin() + axis);
    Tensor result(result_shape, 0.0f);
    for (size_t i = 0; i < size_; i++) {
        std::vector<size_t> coord = IndexToCoord(i, shape_);
        std::vector<size_t> result_coord = coord;
        result_coord[axis] = 0;
        size_t ind = CoordToIndex(result_coord, result.shape_);
        result.data_[ind] += data_[i];
    }
    return result;
}

Tensor Tensor::Mean(int axis) {
    size_t dim = shape_[axis];
    Tensor result = SumAxis(axis);
    result /= dim;
    return result;
}