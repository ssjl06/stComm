# Test configuration

# Find Google Test
if(DEFINED ENV{GTEST_HOME})
    set(GTEST_ROOT "$ENV{GTEST_HOME}")
endif()

find_package(GTest REQUIRED)

# Enable testing
enable_testing()

# Test source files
set(TEST_SOURCES
    test_mpi_comm.cpp
    test_nccl_comm.cpp
    test_integration.cpp
)

# Create test executable
add_executable(stComm_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(stComm_tests
    PRIVATE
        stComm
        GTest::gtest
        GTest::gtest_main
        MPI::MPI_CXX
        CUDA::cudart
)

# Set output directory
set_target_properties(stComm_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Add test
add_test(NAME stComm_tests COMMAND stComm_tests)

# Set test environment
set_tests_properties(stComm_tests PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
)
