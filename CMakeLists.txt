cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(stComm VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find required packages
# MPI
find_package(MPI REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# NCCL - use environment variable if available
if(DEFINED ENV{NCCL_HOME})
    set(NCCL_INCLUDE_DIR "$ENV{NCCL_HOME}/include")
    set(NCCL_LIBRARY "$ENV{NCCL_HOME}/lib/libnccl.so")
else()
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h)
    find_library(NCCL_LIBRARY NAMES nccl)
endif()

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL not found. Please set NCCL_HOME environment variable.")
endif()

message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")

# CUDA architecture
if(DEFINED ENV{CUDA_ARCH})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCH})
else()
    set(CMAKE_CUDA_ARCHITECTURES "75;80;89")
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIR}
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Add subdirectories
add_subdirectory(src/stComm)
add_subdirectory(tests)
