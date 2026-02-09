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

# Find MPI
find_package(MPI REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find NCCL
if(DEFINED ENV{NCCL_HOME})
    set(NCCL_INCLUDE_DIR "$ENV{NCCL_HOME}/include")
    # Try both lib and lib64
    find_library(NCCL_LIBRARY
        NAMES nccl
        PATHS "$ENV{NCCL_HOME}/lib" "$ENV{NCCL_HOME}/lib64"
        NO_DEFAULT_PATH
    )
else()
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h)
    find_library(NCCL_LIBRARY NAMES nccl)
endif()

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL not found. Set NCCL_HOME environment variable or ensure NCCL is in system path.")
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
add_subdirectory(examples)

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Install header files
install(DIRECTORY include/stComm
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Install library (configured in src/stComm/CMakeLists.txt)
# Library targets are exported there

# Generate and install CMake config files
include(CMakePackageConfigHelpers)

# Create stCommConfig.cmake
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/stCommConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/stCommConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stComm
)

# Create stCommConfigVersion.cmake
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/stCommConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install CMake config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/stCommConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/stCommConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stComm
)

# Install export targets
install(EXPORT stCommTargets
    FILE stCommTargets.cmake
    NAMESPACE stComm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stComm
)

# Print installation info
message(STATUS "")
message(STATUS "========================================")
message(STATUS "stComm installation configuration:")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Headers: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/stComm")
message(STATUS "  Library: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
message(STATUS "  CMake config: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/stComm")
message(STATUS "========================================")
message(STATUS "")
