cmake_minimum_required(VERSION 3.24)
project(geo_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find Python (the same interpreter that has torch + numpy installed).
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11: prefer the pip-installed one. Get its cmake dir via:
#   python3 -m pybind11 --cmakedir
# and pass -Dpybind11_DIR=... , or install pybind11 via your package manager.
find_package(pybind11 CONFIG REQUIRED)

add_executable(geo_bridge cpp/main.cpp)
target_include_directories(geo_bridge PRIVATE cpp)
target_link_libraries(geo_bridge PRIVATE pybind11::embed Python3::Python)

# The schedule runner: executes a compiled .ops stream against PyTorch.
add_executable(run_schedule cpp/run_schedule.cpp)
target_include_directories(run_schedule PRIVATE cpp)
target_link_libraries(run_schedule PRIVATE pybind11::embed Python3::Python)

# On macOS with Homebrew LLVM you may want to point CMake at that clang++:
#   cmake -S . -B build \
#     -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
#     -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
