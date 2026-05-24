cmake_minimum_required(VERSION 3.15)
project(bpe_tokenizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(
    unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG v4.4.0
)
FetchContent_MakeAvailable(unordered_dense)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
nanobind_add_module(bpe_tokenizer bpe_tokenizer.cpp)
target_link_libraries(bpe_tokenizer PRIVATE unordered_dense::unordered_dense)
