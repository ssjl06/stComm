# Examples configuration

# MPI examples
add_executable(mpi_hello mpi_hello.cpp)
enable_compiler_warnings(mpi_hello)
target_link_libraries(mpi_hello PRIVATE stComm MPI::MPI_CXX)
set_target_properties(mpi_hello PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

add_executable(mpi_ring mpi_ring.cpp)
enable_compiler_warnings(mpi_ring)
target_link_libraries(mpi_ring PRIVATE stComm MPI::MPI_CXX)
set_target_properties(mpi_ring PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

add_executable(mpi_collective mpi_collective.cpp)
enable_compiler_warnings(mpi_collective)
target_link_libraries(mpi_collective PRIVATE stComm MPI::MPI_CXX)
set_target_properties(mpi_collective PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

add_executable(mpi_large_data mpi_large_data.cpp)
enable_compiler_warnings(mpi_large_data)
target_link_libraries(mpi_large_data PRIVATE stComm MPI::MPI_CXX)
set_target_properties(mpi_large_data PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

# NCCL examples
add_executable(nccl_ring nccl_ring.cpp)
enable_compiler_warnings(nccl_ring)
target_link_libraries(nccl_ring PRIVATE stComm MPI::MPI_CXX CUDA::cudart)
set_target_properties(nccl_ring PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

add_executable(nccl_collective nccl_collective.cpp)
enable_compiler_warnings(nccl_collective)
target_link_libraries(nccl_collective PRIVATE stComm MPI::MPI_CXX CUDA::cudart)
set_target_properties(nccl_collective PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)
