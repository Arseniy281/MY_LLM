#include "../Tensor/tensor.h"
#include "matmul.h"
#include <iostream>
#include <cassert>
#include <cmath>

void TestMatMulNaive() {
    std::cout << "Testing MatMul (naive)...\n";
    
    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f});
    Tensor B({3, 2}, {7.0f, 8.0f,
                      9.0f, 10.0f,
                      11.0f, 12.0f});
    
    Tensor C = MatMul(A, B);
    
    assert(C.GetShape()[0] == 2);
    assert(C.GetShape()[1] == 2);
    assert(std::abs(C.at({0, 0}) - 58.0f) < 1e-5f);
    assert(std::abs(C.at({0, 1}) - 64.0f) < 1e-5f);
    assert(std::abs(C.at({1, 0}) - 139.0f) < 1e-5f);
    assert(std::abs(C.at({1, 1}) - 154.0f) < 1e-5f);
    
    std::cout << "  ✅ MatMul OK\n";
}

void TestMatMulLoopTiling() {
    std::cout << "Testing MatMulLoopTiling...\n";
    
    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f});
    Tensor B({3, 2}, {7.0f, 8.0f,
                      9.0f, 10.0f,
                      11.0f, 12.0f});
    
    Tensor C = MatMulLoopTiling(A, B, 2);
    
    assert(C.GetShape()[0] == 2);
    assert(C.GetShape()[1] == 2);
    assert(std::abs(C.at({0, 0}) - 58.0f) < 1e-5f);
    assert(std::abs(C.at({0, 1}) - 64.0f) < 1e-5f);
    assert(std::abs(C.at({1, 0}) - 139.0f) < 1e-5f);
    assert(std::abs(C.at({1, 1}) - 154.0f) < 1e-5f);
    
    std::cout << "  ✅ MatMulLoopTiling OK\n";
}

void TestMatMulMultithreaded() {
    std::cout << "Testing MatMulMultithreaded...\n";
    
    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f});
    Tensor B({3, 2}, {7.0f, 8.0f,
                      9.0f, 10.0f,
                      11.0f, 12.0f});
    
    Tensor C = MatMulMultithreaded(A, B);
    
    assert(C.GetShape()[0] == 2);
    assert(C.GetShape()[1] == 2);
    assert(std::abs(C.at({0, 0}) - 58.0f) < 1e-5f);
    assert(std::abs(C.at({0, 1}) - 64.0f) < 1e-5f);
    assert(std::abs(C.at({1, 0}) - 139.0f) < 1e-5f);
    assert(std::abs(C.at({1, 1}) - 154.0f) < 1e-5f);
    
    std::cout << "  ✅ MatMulMultithreaded OK\n";
}

void TestAllVersionsMatch() {
    std::cout << "Testing all versions match...\n";
    
    Tensor A = Tensor::Random({8, 6}, -10.0f, 10.0f);
    Tensor B = Tensor::Random({6, 7}, -10.0f, 10.0f);
    
    Tensor C1 = MatMul(A, B);
    Tensor C2 = MatMulLoopTiling(A, B);
    Tensor C3 = MatMulMultithreaded(A, B);
    
    for (size_t i = 0; i < C1.GetSize(); ++i) {
        assert(std::abs(C1.at(i) - C2.at(i)) < 1e-5f);
        assert(std::abs(C1.at(i) - C3.at(i)) < 1e-5f);
    }
    
    std::cout << "  ✅ All versions match\n";
}

int main() {
    std::cout << "\n=== MatMul Tests ===\n\n";
    TestMatMulNaive();
    TestMatMulLoopTiling();
    TestMatMulMultithreaded();
    TestAllVersionsMatch();
    std::cout << "\n✅ All tests passed!\n";
    return 0;
}