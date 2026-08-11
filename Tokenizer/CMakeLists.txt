cmake_minimum_required(VERSION 3.10)
project(Phase6_Tokenizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tokenizer_test
    bpe_tokenizer.cpp
    trie.cpp
    tokenizer_test.cpp
)

target_include_directories(tokenizer_test PRIVATE .)
