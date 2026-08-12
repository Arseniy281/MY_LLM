#include "tensor.h"
#include <vector>
#include <cmath>

std::vector<size_t> AllignTensors(const std::vector<size_t>& small, const std::vector<size_t>& big) {
    if (small.size() > big.size()) { return small; }
    size_t final_size = big.size();
    if (small.size() == big.size()) {
        for (size_t i = 0; i < big.size(); i++) {
            if (small[i] != big[i] && !(small[i] == 1 || big[i] == 1)) {
                throw std::runtime_error("You cannot Broadcst this matrixes");
            }
        }
        return small;
    } else {
        std::vector<size_t> new_small;
        size_t i = 0;
        size_t j = 0;
        while (j < big.size()) {
            if (small[i] == big[j] || small[i] == 1) {
                new_small.push_back(small[i]);
                i++;
                j++;
            } else {
                new_small.push_back(1);
                j++;
            }
        }
        if (new_small.size() > final_size || j != big.size() || i != small.size()) {
            throw std::runtime_error("You cannot Broadcst this matrixes");
        }
        return new_small;
    }
}

std::vector<size_t> GetFinalShape(const std::vector<size_t>& first, const std::vector<size_t>& second) {
    if (first.size() != second.size()) {
        throw std::runtime_error("You cannot Broadcst this matrixes");
    }
    std::vector<size_t> final_shape(first.size());
    for (size_t i = 0; i < first.size(); i++) {
        final_shape[i] = std::max(first[i], second[i]);
    }
    return final_shape;
}

size_t GetFinalSize(const std::vector<size_t>& final_shape) {
    size_t final_size = 1;
    for (const auto& dim : final_shape) {
        final_size *= dim;
    }
    return final_size;
}

std::vector<size_t> IndexToCoord(size_t ind, const std::vector<size_t>& shape) {
    std::vector<size_t> coords(shape.size());
    for (int i = shape.size() - 1; i >= 0; i--) {
        coords[i] = ind % shape[i];
        ind /= shape[i];
    }
    return coords;
}

size_t CoordToIndex(const std::vector<size_t>& coord, const std::vector<size_t>& shape) {
    size_t index = 0;
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        index += coord[i] * stride;
        stride *= shape[i];
    }
    return index;
}

size_t BroadcastIndex(const std::vector<size_t>& real_shape, const std::vector<size_t>& final_shape, size_t ind) {
    std::vector<size_t> coords = IndexToCoord(ind, final_shape);
    for (size_t i = 0; i < final_shape.size(); i++) {
        if (real_shape[i] == 1) {
            coords[i] = 0;
        }
    }
    size_t real_index = CoordToIndex(coords, real_shape);
    return real_index;
}

Tensor Tensor::operator+(const Tensor& other) const {
    std::vector<size_t> s1 = shape_;
    std::vector<size_t> s2 = other.GetShape();
    s1 = AllignTensors(s1, s2);
    s2 = (AllignTensors(s2, s1));

    const float* data_1 = RawData();
    const float* data_2 = other.RawData();

    std::vector<size_t> final_shape = GetFinalShape(s1, s2);
    size_t final_size = GetFinalSize(final_shape);

    std::vector<float> final_data(final_size);
    for (size_t i = 0; i < final_size; i++) {
        final_data[i] = data_1[BroadcastIndex(s1, final_shape, i)] 
                        + data_2[BroadcastIndex(s2, final_shape, i)];
    }

    Tensor result(final_shape, final_data);
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