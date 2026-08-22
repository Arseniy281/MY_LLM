#include "tensor.h"
#include <vector>
#include <cmath>

std::vector<size_t> Tensor::AlignTensors(const std::vector<size_t>& small, const std::vector<size_t>& big) {
    if (small.empty()) { return std::vector<size_t>(big.size(), 1); }
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
            if (i < small.size() && (small[i] == big[j] || small[i] == 1)) {
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

std::vector<size_t> Tensor::GetFinalShape(const std::vector<size_t>& first, const std::vector<size_t>& second) {
    if (first.size() != second.size()) {
        throw std::runtime_error("You cannot Broadcast this matrixes");
    }
    std::vector<size_t> final_shape(first.size());
    for (size_t i = 0; i < first.size(); i++) {
        final_shape[i] = std::max(first[i], second[i]);
    }
    return final_shape;
}

size_t Tensor::GetFinalSize(const std::vector<size_t>& final_shape) {
    size_t final_size = 1;
    for (const auto& dim : final_shape) {
        final_size *= dim;
    }
    return final_size;
}

std::vector<size_t> Tensor::IndexToCoord(size_t ind, const std::vector<size_t>& shape) {
    std::vector<size_t> coords(shape.size());
    for (int i = shape.size() - 1; i >= 0; i--) {
        coords[i] = ind % shape[i];
        ind /= shape[i];
    }
    return coords;
}

size_t Tensor::CoordToIndex(const std::vector<size_t>& coord, const std::vector<size_t>& shape) {
    size_t index = 0;
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        index += coord[i] * stride;
        stride *= shape[i];
    }
    return index;
}

size_t Tensor::BroadcastIndex(const std::vector<size_t>& real_shape, const std::vector<size_t>& final_shape, size_t ind) {
    std::vector<size_t> coords = Tensor::IndexToCoord(ind, final_shape);
    for (size_t i = 0; i < final_shape.size(); i++) {
        if (real_shape[i] == 1) {
            coords[i] = 0;
        }
    }
    size_t real_index = Tensor::CoordToIndex(coords, real_shape);
    return real_index;
}

Tensor Tensor::operator+(const Tensor& other) const {
    std::vector<size_t> s1 = shape_;
    std::vector<size_t> s2 = other.GetShape();
    s1 = AlignTensors(s1, s2);
    s2 = (AlignTensors(s2, s1));

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
    *this = *this + other;
    return *this;
}

Tensor Tensor::operator-(const Tensor& other) const {
    std::vector<size_t> s1 = shape_;
    std::vector<size_t> s2 = other.GetShape();
    s1 = AlignTensors(s1, s2);
    s2 = (AlignTensors(s2, s1));

    const float* data_1 = RawData();
    const float* data_2 = other.RawData();

    std::vector<size_t> final_shape = GetFinalShape(s1, s2);
    size_t final_size = GetFinalSize(final_shape);

    std::vector<float> final_data(final_size);
    for (size_t i = 0; i < final_size; i++) {
        final_data[i] = data_1[BroadcastIndex(s1, final_shape, i)] 
                        - data_2[BroadcastIndex(s2, final_shape, i)];
    }

    Tensor result(final_shape, final_data);
    return result;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    *this = *this - other;
    return *this;
}

Tensor Tensor::operator*(const Tensor& other) const {
    std::vector<size_t> s1 = shape_;
    std::vector<size_t> s2 = other.GetShape();
    s1 = AlignTensors(s1, s2);
    s2 = (AlignTensors(s2, s1));

    const float* data_1 = RawData();
    const float* data_2 = other.RawData();

    std::vector<size_t> final_shape = GetFinalShape(s1, s2);
    size_t final_size = GetFinalSize(final_shape);

    std::vector<float> final_data(final_size);
    for (size_t i = 0; i < final_size; i++) {
        final_data[i] = data_1[BroadcastIndex(s1, final_shape, i)] 
                        * data_2[BroadcastIndex(s2, final_shape, i)];
    }

    Tensor result(final_shape, final_data);
    return result;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other;
    return *this;
}

Tensor Tensor::operator/(const Tensor& other) const {
    std::vector<size_t> s1 = shape_;
    std::vector<size_t> s2 = other.GetShape();
    s1 = AlignTensors(s1, s2);
    s2 = (AlignTensors(s2, s1));

    const float* data_1 = RawData();
    const float* data_2 = other.RawData();

    std::vector<size_t> final_shape = GetFinalShape(s1, s2);
    size_t final_size = GetFinalSize(final_shape);

    std::vector<float> final_data(final_size);
    for (size_t i = 0; i < final_size; i++) {
        size_t idx1 = BroadcastIndex(s1, final_shape, i);
        size_t idx2 = BroadcastIndex(s2, final_shape, i);
        
        float divisor = data_2[idx2];
        if (divisor == 0.0f) {
            final_data[i] = 0.0f;
        } else {
            final_data[i] = data_1[idx1] / divisor;
        }
    }

    Tensor result(final_shape, final_data);
    return result;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other;
    return *this;
}

Tensor Tensor::operator+(const float num) const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < size_; i++) {
        tensor.data_[i] = data_[i] + num;
    }
    return tensor;
}

Tensor& Tensor::operator+=(const float num) {
    for (size_t i = 0; i < size_; i++) {
        data_[i] += num;
    }
    return *this;
}

Tensor Tensor::operator-(const float num) const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < size_; i++) {
        tensor.data_[i] = data_[i] - num;
    }
    return tensor;
}

Tensor& Tensor::operator-=(const float num) {
    for (size_t i = 0; i < size_; i++) {
        data_[i] -= num;
    }
    return *this;
}

Tensor Tensor::operator*(const float num) const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < size_; i++) {
        tensor.data_[i] = data_[i] * num;
    }
    return tensor;
}

Tensor& Tensor::operator*=(const float num) {
    for (size_t i = 0; i < size_; i++) {
        data_[i] *= num;
    }
    return *this;
}

Tensor Tensor::operator/(const float num) const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < size_; i++) {
        if (num == 0) {
            throw std::runtime_error("Division by zero");
        }
        tensor.data_[i] = data_[i] / num;
    }
    return tensor;
}

Tensor& Tensor::operator/=(const float num) {
    for (size_t i = 0; i < size_; i++) {
        if (num == 0) {
            throw std::runtime_error("Division by zero");
        } 
        data_[i] /= num;
    }
    return *this;
}

Tensor Tensor::operator-() const {
    Tensor tensor(shape_);
    for (size_t i = 0; i < size_; i++) {
        tensor.data_[i] = -data_[i];
    }
    return tensor;
}

void Tensor::CheckBeforeConcatenate(const std::vector<Tensor>& tensors, size_t axis) {
    if (tensors.size() == 0) {
        throw std::runtime_error("Tensors must have size > 0");
    }
    std::vector<size_t> cur_shape = tensors[0].GetShape();
    if (axis >= cur_shape.size()) {
        throw std::runtime_error("Axis out of bounds");
    }
    std::vector<size_t> last_shape = cur_shape;
    size_t num_heads = tensors.size();
    for (size_t i = 1; i < num_heads; i++) {
        cur_shape = tensors[i].GetShape();
        if (last_shape.size() != cur_shape.size()) {
            throw std::runtime_error("All tensors must have same rank");
        }
        for (size_t j = 0; j < last_shape.size(); j++) {
            if (axis != j && last_shape[j] != cur_shape[j]) {
                throw std::runtime_error("All tensors must have same shape on axis " + std::to_string(axis));
            }
        }
        last_shape = cur_shape;
    }
}

Tensor Tensor::Concatenate(const std::vector<Tensor>& tensors, size_t axis) {
    if (tensors.empty()) {
        throw std::runtime_error("Tensors must have size > 0");
    }
    
    std::vector<size_t> final_shape = tensors[0].GetShape();
    for (size_t t = 1; t < tensors.size(); t++) {
        auto shape = tensors[t].GetShape();
        if (shape.size() != final_shape.size()) {
            throw std::runtime_error("All tensors must have same rank");
        }
        for (size_t j = 0; j < final_shape.size(); j++) {
            if (j != axis && final_shape[j] != shape[j]) {
                throw std::runtime_error("All tensors must have same shape except on axis");
            }
        }
    }
    
    size_t total_dim = 0;
    for (const auto& t : tensors) {
        total_dim += t.GetShape()[axis];
    }
    final_shape[axis] = total_dim;
    
    Tensor output(final_shape);
    float* result_data = output.RawData();
    size_t offset = 0;
    
    for (const auto& tensor : tensors) {
        const float* data = tensor.RawData();
        size_t tensor_size = tensor.GetSize();
        
        for (size_t i = 0; i < tensor_size; i++) {
            std::vector<size_t> coords = IndexToCoord(i, tensor.GetShape());
            coords[axis] += offset;
            size_t idx = CoordToIndex(coords, final_shape);
            result_data[idx] = data[i];
        }
        offset += tensor.GetShape()[axis];
    }
    
    return output;
}

Tensor Tensor::Transpose() {
    std::vector<size_t> new_shape = shape_;
    size_t shape_size = new_shape.size();
    std::swap(new_shape[shape_size - 1], new_shape[shape_size - 2]);
    Tensor tensor(new_shape);
    
    std::vector<size_t> cur_coords(shape_size, 0);
    std::vector<size_t> new_coords(shape_size, 0);
    
    for (size_t i = 0; i < shape_[shape_size - 2]; i++) {
        for (size_t j = 0; j < shape_[shape_size - 1]; j++) {
            cur_coords[shape_size - 2] = i;
            cur_coords[shape_size - 1] = j;
            new_coords[shape_size - 2] = j;
            new_coords[shape_size - 1] = i;
            tensor.at(new_coords) = this->at(cur_coords);
        }
    }
    return tensor;
}


Tensor operator/(float scalar, const Tensor& t) {
    Tensor result(t.GetShape());
    for (size_t i = 0; i < t.GetSize(); i++) {
        result.RawData()[i] = scalar / t.RawData()[i];
    }
    return result;
}