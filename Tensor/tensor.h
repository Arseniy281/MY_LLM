#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include "../Autograd/operation.h"

static const float EPSILON = 1e-8f;


class Tensor {
private:
    std::vector<size_t> shape_ = {};
    std::vector<float> data_ = {};
    size_t size_ = 0;
    size_t rank_ = 0;

    Tensor* grad_ = nullptr;
    Operation* grad_fn_ = nullptr;

    size_t ComputeIndex(const std::vector<size_t>& indexes) const;
public:

    friend class AddOp;
    friend class SubOp;
    friend class MulOp;
    friend class SquareOp;
    friend class SumOp;

    explicit Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, float k);
    static Tensor Random(std::vector<size_t> shape, float min = 0.0f, float max = 1.0f);
    Tensor(std::vector<size_t> shape, std::vector<float> data);
    Tensor() = default;


    float& at(size_t index);
    const float& at(size_t index) const;
    float& at(std::vector<size_t> indexes);
    const float& at(std::vector<size_t> indexes) const;
    void Set(std::vector<size_t> indexes, float value);
    auto GetIter(const std::vector<size_t>& indexes);
    const auto GetIter(const std::vector<size_t>& indexes) const;

    Tensor operator+(const Tensor& other) const;
    Tensor& operator+=(const Tensor& other);
    Tensor operator-(const Tensor& other) const;
    Tensor& operator-=(const Tensor& other);
    Tensor operator*(const Tensor& other) const;
    Tensor& operator*=(const Tensor& other);
    Tensor operator/(const Tensor& other) const;
    Tensor& operator/=(const Tensor& other);

    Tensor operator+(const float num) const;
    Tensor& operator+=(const float num);
    Tensor operator-(const float num) const;
    Tensor& operator-=(const float num);
    Tensor operator*(const float num) const;
    Tensor& operator*=(const float num);
    Tensor operator/(const float num) const;
    Tensor& operator/=(const float num);
    Tensor operator-() const;

    Tensor Transpose();
    void Reshape(std::vector<size_t> new_shape);
    Tensor Reshape(std::vector<size_t> new_shape) const;
    size_t GetSize() const;
    size_t GetRank() const;
    const std::vector<size_t>& GetShape() const;
    float* data();
    const float* data() const;

    void print() const;
    void PrintInfo() const;

    Tensor* Grad() const;
    void PrintGrad() const;
    Operation* GradFn() const;
    void backward(const Tensor& grad_output = Tensor({1, 1}, 1.0f));
};