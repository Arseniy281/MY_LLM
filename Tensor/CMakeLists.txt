cmake_minimum_required(VERSION 3.10)
project(Phase1_Tensor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tensor_tests 
    tensor_tests.cpp 
    tensor.cpp
    tensor_ops.cpp
    tensor_print.cpp
)

add_executable(broadcast_test
    broadcast_test.cpp 
    tensor.cpp
    tensor_ops.cpp
    tensor_print.cpp
)

target_include_directories(tensor_tests PRIVATE .)
target_include_directories(broadcast_test PRIVATE .)