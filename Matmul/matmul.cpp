#include "matmul.h"
#include <vector>
#include <thread>

Tensor MatMul(const Tensor& first, const Tensor& second) {
    if (first.GetRank() != 2 || second.GetRank() != 2) {
        throw std::runtime_error("You can multiply only matrix");
    }
    const std::vector<size_t> first_shape = first.GetShape();
    const std::vector<size_t> second_shape = second.GetShape();

    size_t M = first_shape[0];
    size_t K = first_shape[1];
    size_t N = second_shape[1];

    const float* first_data = first.data();
    const float* second_data = second.data();

    Tensor tensor({M, N});
    float* tensor_data = tensor.data();

    if (first_shape[1] != second_shape[0]) {
        throw std::runtime_error("You cannot multiply this matrixes");
    }

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float val = 0;
            for (size_t k = 0; k < K; k++) {
                val += first_data[i * K + k] * second_data[k * N + j];
            } 
            tensor_data[i * N + j] = val;
        }
    }
    return tensor;
}

Tensor MatMulMultithreaded(const Tensor& first, const Tensor& second) {
    if (first.GetRank() != 2 || second.GetRank() != 2) {
        throw std::runtime_error("You can multiply only matrix");
    }
    const std::vector<size_t> first_shape = first.GetShape();
    const std::vector<size_t> second_shape = second.GetShape();

    if (first_shape[1] != second_shape[0]) {
        throw std::runtime_error("You cannot multiply this matrixes");
    }

    size_t M = first_shape[0];
    size_t K = first_shape[1];
    size_t N = second_shape[1];

    const float* first_data = first.data();
    const float* second_data = second.data();

    Tensor tensor({M, N});
    float* tensor_data = tensor.data();

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) { num_threads = 4; }

    std::vector<std::thread> threads;

    size_t rows_per_thread = M / num_threads;
    size_t remaining_rows = M % num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = (t + 1) * rows_per_thread;
        if (t == num_threads - 1) { end_row = M; };
        if (start_row >= M) break;

        threads.emplace_back([start_row, end_row, &first_data, 
            &second_data, &tensor_data, N, K](){

            for (size_t i = start_row; i < end_row; i++) {
                for (size_t j = 0; j < N; j++) {
                    float val = 0.0f;
                    for (size_t k = 0; k < K; k++) {
                        val += first_data[i * K + k] * second_data[k * N + j];
                    } 
                    tensor_data[i * N + j] = val;
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    return tensor;
}

Tensor MatMulLoopTiling(const Tensor& first, const Tensor& second, size_t block_size) {
    if (first.GetRank() != 2 || second.GetRank() != 2) {
        throw std::runtime_error("You can multiply only matrix");
    }
    const std::vector<size_t> first_shape = first.GetShape();
    const std::vector<size_t> second_shape = second.GetShape();

    if (first_shape[1] != second_shape[0]) {
        throw std::runtime_error("You cannot multiply this matrixes");
    }

    size_t M = first_shape[0];
    size_t K = first_shape[1];
    size_t N = second_shape[1];

    const float* first_data = first.data();
    const float* second_data = second.data();

    Tensor tensor({M, N});
    float* tensor_data = tensor.data();

    for (size_t i_block = 0; i_block < M; i_block += block_size) {
        size_t i_end = std::min(i_block + block_size, M);
        for (size_t j_block = 0; j_block < N; j_block += block_size) {
            size_t j_end = std::min(j_block + block_size, N);
            for (size_t k_block = 0; k_block < K; k_block += block_size) {
                size_t k_end = std::min(k_block + block_size, K);

                for (size_t i = i_block; i < i_end; i++) {
                    for (size_t j = j_block; j < j_end; j++) {
                        float val = 0.0f;
                        for (size_t k = k_block; k < k_end; k++) {
                            val += first_data[i * K + k] * second_data[k * N + j];
                        }
                        tensor_data[i * N + j] += val;
                    }
                }
            }
        }
    }
    return tensor;
}