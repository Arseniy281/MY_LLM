#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <memory>
#include "../Autograd/operation.h"

static const float EPSILON = 1e-8f;


class Tensor {
private:
    std::vector<size_t> shape_ = {};
    std::vector<float> data_ = {};
    size_t size_ = 0;
    size_t rank_ = 0;

    std::shared_ptr<Tensor> grad_ = nullptr;
    Operation* grad_fn_ = nullptr;

    size_t ComputeIndex(const std::vector<size_t>& indexes) const;
    static void CheckBeforeConcatenate(const std::vector<Tensor>& head_outputs, size_t axis = 0);
public:

    explicit Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, float k);
    static Tensor Random(std::vector<size_t> shape, float min = 0.0f, float max = 1.0f);
    Tensor(std::vector<size_t> shape, std::vector<float> data);
    Tensor() = default;


    float& at(size_t index);
    const float& at(size_t index) const;
    float& at(const std::vector<size_t>& indexes);
    const float& at(const std::vector<size_t>& indexes) const;
    void Set(std::vector<size_t> indexes, float value);
    auto GetIter(const std::vector<size_t>& indexes);
    const auto GetIter(const std::vector<size_t>& indexes) const;

    static std::vector<size_t> IndexToCoord(size_t ind, const std::vector<size_t>& shape);
    static size_t CoordToIndex(const std::vector<size_t>& coord, const std::vector<size_t>& shape);

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
    static Tensor Concatenate(const std::vector<Tensor>& head_outputs, size_t axis = 0);
    static size_t BroadcastIndex(const std::vector<size_t>& real_shape, const std::vector<size_t>& final_shape, size_t ind);
    static std::vector<size_t> AlignTensors(const std::vector<size_t>& small, const std::vector<size_t>& big);
    static std::vector<size_t> GetFinalShape(const std::vector<size_t>& first, const std::vector<size_t>& second);
    static size_t GetFinalSize(const std::vector<size_t>& final_shape);

    Tensor Transpose();
    void Reshape(std::vector<size_t> new_shape);
    Tensor Reshape(std::vector<size_t> new_shape) const;
    size_t GetSize() const;
    size_t GetRank() const;
    const std::vector<size_t>& GetShape() const;
    std::vector<float> Data();
    const std::vector<float> Data() const;
    float* RawData();
    const float* RawData() const;

    void print() const;
    void PrintInfo() const;

    std::shared_ptr<Tensor> Grad() const;
    void PrintGrad() const;
    void ClearGrad();
    Operation* GradFn() const;
    void SetGradFn(Operation* op);
    void backward(const Tensor& grad_output = Tensor({1, 1}, 1.0f));

    Tensor SumAxis(int axis);
    Tensor Mean(int axis);
};

Tensor operator/(float scalar, const Tensor& t);