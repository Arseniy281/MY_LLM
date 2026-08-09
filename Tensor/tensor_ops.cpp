#include "tensor.h"

bool IsBroadcastCompatible(const std::vector<size_t>& s1, const std::vector<size_t>& s2) {
    auto to2D = [](const std::vector<size_t>& s) {
        if (s.size() == 1) { return std::vector<size_t>{1, s[0]}; }
        if (s.size() == 2) { return s; }
        throw std::runtime_error("Broadcast only supports 1D and 2D tensors");
    };
    
    std::vector<size_t> a = to2D(s1);
    std::vector<size_t> b = to2D(s2);
    
    if (a[0] != b[0] && a[0] != 1 && b[0] != 1) { return false; }
    if (a[1] != b[1] && a[1] != 1 && b[1] != 1) { return false; }
    
    return true;
}

Tensor Tensor::operator+(const Tensor& other) const { 
    if (shape_.size() > 2 && other.shape_.size() > 2) {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shapes must match for >2D tensors");
        }
        Tensor result(shape_);
        for (size_t i = 0; i < size_; i++) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    auto to2D = [](const Tensor& t) {
        if (t.shape_.size() == 1) {
            return std::vector<size_t>{1, t.shape_[0]};
        }
        return t.shape_;
    };
    
    std::vector<size_t> s1 = to2D(*this);
    std::vector<size_t> s2 = to2D(other);

    if (!IsBroadcastCompatible(s1, s2)) {
        throw std::runtime_error("Different shapes for summation"); 
    }

    std::vector<size_t> final_shape = {std::max(s1[0], s2[0]), std::max(s1[1], s2[1])};
    Tensor result(final_shape);
    
    for (size_t i = 0; i < final_shape[0]; i++) {
        for (size_t j = 0; j < final_shape[1]; j++) {
            size_t i1 = (s1[0] == 1) ? 0 : i;
            size_t j1 = (s1[1] == 1) ? 0 : j;
            
            size_t i2 = (s2[0] == 1) ? 0 : i;
            size_t j2 = (s2[1] == 1) ? 0 : j;
            
            result.at({i, j}) = at({i1, j1}) + other.at({i2, j2});
        }
    }
    return result;
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