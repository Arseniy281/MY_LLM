#include "tensor.h"

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Different shapes for summation"); 
    }
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        tensor.data_[i] = data_[i] + other.data_[i];
    }
    return tensor;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Different shapes for summation"); 
    }
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Different shapes for subtraction"); 
    }
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        tensor.data_[i] = data_[i] - other.data_[i];
    }
    return tensor;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Different shapes for subtraction"); 
    }
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Different shapes for multiplication"); 
    }
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        tensor.data_[i] = data_[i] * other.data_[i];
    }
    return tensor;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Different shapes for multiplication"); 
    }
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] *= other.data_[i];
    }
    return *this;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Different shapes for division"); 
    }
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        if (other.data_[i] == 0) {
            tensor.data_[i] = data_[i] / (other.data_[i] + EPSILON);
        } else {
            tensor.data_[i] = data_[i] / other.data_[i];
        }
    }
    return tensor;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Different shapes for division"); 
    }
    for (size_t i = 0; i < data_.size(); i++) {
        if (other.data_[i] == 0) {
            data_[i] /= (other.data_[i] + EPSILON);
        } else {
            data_[i] /= other.data_[i];
        }
    }
    return *this;
}

Tensor Tensor::operator+(const float num) const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        tensor.data_[i] = data_[i] + num;
    }
    return tensor;
}

Tensor& Tensor::operator+=(const float num) {
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] += num;
    }
    return *this;
}

Tensor Tensor::operator-(const float num) const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        tensor.data_[i] = data_[i] - num;
    }
    return tensor;
}

Tensor& Tensor::operator-=(const float num) {
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] -= num;
    }
    return *this;
}

Tensor Tensor::operator*(const float num) const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        tensor.data_[i] = data_[i] * num;
    }
    return tensor;
}

Tensor& Tensor::operator*=(const float num) {
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] *= num;
    }
    return *this;
}

Tensor Tensor::operator/(const float num) const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        if (num == 0) {
            throw std::runtime_error("Division by zero");
        }
        tensor.data_[i] = data_[i] / num;
    }
    return tensor;
}

Tensor& Tensor::operator/=(const float num) {
    for (size_t i = 0; i < data_.size(); i++) {
        if (num == 0) {
            throw std::runtime_error("Division by zero");
        } 
        data_[i] /= num;
    }
    return *this;
}

Tensor Tensor::operator-() const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < data_.size(); i++) {
        tensor.data_[i] = -data_[i];
    }
    return tensor;
}