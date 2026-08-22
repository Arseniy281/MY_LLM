#include "matmul.h"
#include <vector>
#include <thread>

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

    const float* first_data = first.RawData();
    const float* second_data = second.RawData();

    Tensor tensor({M, N});
    float* tensor_data = tensor.RawData();

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

    const float* first_data = first.RawData();
    const float* second_data = second.RawData();

    Tensor tensor({M, N});
    float* tensor_data = tensor.RawData();

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

std::vector<size_t> AlignTensorsForMatMul(const std::vector<size_t>& small, const std::vector<size_t>& big) {
    std::vector<size_t> batch1 = small;
    batch1.pop_back();
    batch1.pop_back();
    
    std::vector<size_t> batch2 = big;
    batch2.pop_back();
    batch2.pop_back();

    std::vector<size_t> aligned_batch;
    if (batch1.size() > batch2.size()) {
        aligned_batch = batch1;
    } else if (batch1.size() == batch2.size()) {
        for (size_t i = 0; i < batch2.size(); i++) {
            if (batch1[i] != batch2[i] && !(batch1[i] == 1 || batch2[i] == 1)) {
                throw std::runtime_error("Incompatible batch shapes for MatMul");
            }
        }
        aligned_batch = batch1;
    } else {
        std::vector<size_t> new_small;
        size_t i = 0;
        size_t j = 0;
        while (j < batch2.size()) {
            if (i < batch1.size() && (batch1[i] == batch2[j] || batch1[i] == 1)) {
                new_small.push_back(batch1[i]);
                i++;
                j++;
            } else {
                new_small.push_back(1);
                j++;
            }
        }
        if (i != batch1.size() || j != batch2.size()) {
            throw std::runtime_error("Incompatible batch shapes for MatMul");
        }
        aligned_batch = new_small;
    }

    std::vector<size_t> result = aligned_batch;
    result.push_back(small[small.size() - 2]);
    result.push_back(small[small.size() - 1]);
    return result;
}

Tensor MatMul(const Tensor& A, const Tensor& B) {
    std::vector<size_t> s1 = A.GetShape();
    std::vector<size_t> s2 = B.GetShape();
    
    s1 = AlignTensorsForMatMul(s1, s2);
    s2 = AlignTensorsForMatMul(s2, s1);
    
    if (s1[s1.size() - 1] != s2[s2.size() - 2]) {
        throw std::runtime_error("Incompatible shapes for MatMul");
    }
    
    std::vector<size_t> final_shape = s1;
    final_shape[final_shape.size() - 1] = s2[s2.size() - 1];
    
    Tensor result(final_shape);
    float* result_data = result.RawData();
    
    size_t batch_dims = final_shape.size() - 2;
    size_t M = s1[s1.size() - 2];
    size_t K = s1[s1.size() - 1];
    size_t N = s2[s2.size() - 1];
    
    size_t batch_count = 1;
    for (size_t i = 0; i < batch_dims; i++) {
        batch_count *= final_shape[i];
    }
    
    for (size_t batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        std::vector<size_t> batch_shape = final_shape;
        batch_shape.resize(batch_dims);
        std::vector<size_t> batch_coords = Tensor::IndexToCoord(batch_idx, batch_shape);
        
        size_t offset_A = 0;
        size_t offset_B = 0;
        size_t offset_C = 0;
        size_t stride_A = 1;
        size_t stride_B = 1;
        size_t stride_C = 1;
        
        for (int i = batch_dims - 1; i >= 0; i--) {
            offset_A += batch_coords[i] * stride_A;
            offset_B += batch_coords[i] * stride_B;
            offset_C += batch_coords[i] * stride_C;
            stride_A *= s1[i];
            stride_B *= s2[i];
            stride_C *= final_shape[i];
        }
        
        const float* A_data = A.RawData() + offset_A * M * K;
        const float* B_data = B.RawData() + offset_B * K * N;
        float* C_data = result_data + offset_C * M * N;
        
        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                float a_val = A_data[i * K + k];
                if (a_val == 0.0f) { continue; }
                for (size_t j = 0; j < N; j++) {
                    C_data[i * N + j] += a_val * B_data[k * N + j];
                }
            }
        }
    }
    
    return result;
}