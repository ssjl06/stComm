#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include <iostream>

// Main function for test runner
int main(int argc, char** argv) {
    // Initialize MPI
    stComm::MPIComm::initialize(&argc, &argv);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Get rank
    stComm::MPIComm comm;
    int rank = comm.getRank();

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    stComm::MPIComm::finalize();

    return result;
}

TEST(IntegrationTest, VersionCheck) {
    const char* version = stComm::getVersion();
    EXPECT_NE(version, nullptr);
    EXPECT_STREQ(version, "1.0.0");
}

TEST(IntegrationTest, MPINCCLInterop) {
    stComm::MPIComm mpi_comm;
    int rank = mpi_comm.getRank();
    int size = mpi_comm.getSize();

    // Verify MPI is working
    EXPECT_GE(rank, 0);
    EXPECT_LT(rank, size);

    // Check if CUDA is available for NCCL
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error == cudaSuccess && device_count > 0) {
        // CUDA is available, we could initialize NCCL
        SUCCEED() << "CUDA available with " << device_count << " device(s)";

        // Get NCCL unique ID
        if (rank == 0) {
            ncclUniqueId nccl_id = stComm::NCCLComm::getUniqueId();
            // This verifies that NCCL library is properly linked
            SUCCEED() << "NCCL unique ID generated successfully";
        }
    } else {
        GTEST_SKIP() << "CUDA not available, skipping NCCL interop test";
    }
}

TEST(IntegrationTest, MultipleRequests) {
    stComm::MPIComm comm;
    int rank = comm.getRank();
    int size = comm.getSize();

    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int num_messages = 5;
    const int count = 20;

    std::vector<stComm::RequestPtr> requests;
    std::vector<std::vector<int>> send_buffers(num_messages, std::vector<int>(count));
    std::vector<std::vector<int>> recv_buffers(num_messages, std::vector<int>(count, 0));

    // Initialize send buffers
    for (int msg = 0; msg < num_messages; ++msg) {
        for (int i = 0; i < count; ++i) {
            send_buffers[msg][i] = rank * 1000 + msg * 100 + i;
        }
    }

    // Issue multiple async operations
    if (rank == 0) {
        for (int msg = 0; msg < num_messages; ++msg) {
            auto req = comm.send(send_buffers[msg].data(), count, 1, msg);
            requests.push_back(req);
        }
    } else if (rank == 1) {
        for (int msg = 0; msg < num_messages; ++msg) {
            auto req = comm.recv(recv_buffers[msg].data(), count, 0, msg);
            requests.push_back(req);
        }
    }

    // Wait for all requests
    for (auto& req : requests) {
        EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);
    }

    // Verify received data
    if (rank == 1) {
        for (int msg = 0; msg < num_messages; ++msg) {
            for (int i = 0; i < count; ++i) {
                EXPECT_EQ(recv_buffers[msg][i], msg * 100 + i);
            }
        }
    }
}

TEST(IntegrationTest, LargeDataTransfer) {
    stComm::MPIComm comm;
    int rank = comm.getRank();
    int size = comm.getSize();

    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // Test with large data size (10MB)
    const size_t count = 10 * 1024 * 1024 / sizeof(double);
    std::vector<double> send_data;
    std::vector<double> recv_data;

    if (rank == 0) {
        send_data.resize(count);
        for (size_t i = 0; i < count; ++i) {
            send_data[i] = static_cast<double>(i % 1000) * 0.1;
        }
    } else if (rank == 1) {
        recv_data.resize(count, 0.0);
    }

    stComm::RequestPtr req;

    if (rank == 0) {
        req = comm.send(send_data.data(), count, 1, 0);
    } else if (rank == 1) {
        req = comm.recv(recv_data.data(), count, 0, 0);
    }

    if (req) {
        EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);
    }

    // Spot check some values
    if (rank == 1) {
        EXPECT_DOUBLE_EQ(recv_data[0], 0.0);
        EXPECT_DOUBLE_EQ(recv_data[10], 1.0);
        EXPECT_DOUBLE_EQ(recv_data[999], 99.9);
    }
}
