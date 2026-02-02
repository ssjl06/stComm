/**
 * @file test_nccl_comm.cpp
 * @brief Tests for NCCLComm class
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include <vector>
#include <cuda_runtime.h>

class NCCLCommTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize MPI for coordination
        mpi_comm = std::make_unique<stComm::MPIComm>();
        rank = mpi_comm->getRank();
        size = mpi_comm->getSize();
        device_id = rank % 4; // Assume up to 4 GPUs per node

        // Get NCCL unique ID
        ncclUniqueId nccl_id;
        if (rank == 0) {
            nccl_id = stComm::NCCLComm::getUniqueId();
        }
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Initialize NCCL communicator (creates internal CUDA stream)
        nccl_comm = std::make_unique<stComm::NCCLComm>();
        nccl_comm->initialize(rank, size, device_id, nccl_id);

        cudaSetDevice(device_id);
    }

    void TearDown() override {
        nccl_comm.reset();
        mpi_comm.reset();
    }

    std::unique_ptr<stComm::MPIComm> mpi_comm;
    std::unique_ptr<stComm::NCCLComm> nccl_comm;
    int rank;
    int size;
    int device_id;
};

// Test basic properties
TEST_F(NCCLCommTest, BasicProperties) {
    EXPECT_GE(rank, 0);
    EXPECT_LT(rank, size);
    EXPECT_GT(size, 0);
    EXPECT_EQ(nccl_comm->getBackend(), stComm::Backend::NCCL);
}

// Test send/recv
TEST_F(NCCLCommTest, SendRecv) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 100;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    if (rank == 0) {
        std::vector<float> h_data(N);
        for (int i = 0; i < N; ++i) {
            h_data[i] = i * 1.5f;
        }
        cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        auto req = nccl_comm->send(d_data, N, 1);
        req->wait();
    } else if (rank == 1) {
        auto req = nccl_comm->recv(d_data, N, 0);
        req->wait();

        std::vector<float> h_data(N);
        cudaMemcpy(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; ++i) {
            EXPECT_NEAR(h_data[i], i * 1.5f, 1e-6f);
        }
    }

    cudaFree(d_data);
}

// Test allgatherv with auto displacement
TEST_F(NCCLCommTest, Allgatherv) {
    // Each rank sends different amount
    int sendcount = (rank + 1) * 10;

    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (i + 1) * 10;
    }

    int total_recv = stComm::Utils::totalSize(recvcounts);

    // Allocate device memory
    int *d_sendbuf, *d_recvbuf;
    cudaMalloc(&d_sendbuf, sendcount * sizeof(int));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(int));

    // Initialize send buffer
    std::vector<int> h_sendbuf(sendcount, rank * 100);
    cudaMemcpy(d_sendbuf, h_sendbuf.data(), sendcount * sizeof(int), cudaMemcpyHostToDevice);

    // Call allgatherv (displacement auto-calculated, uses internal stream)
    auto req = nccl_comm->allgatherv(d_sendbuf, sendcount, d_recvbuf, recvcounts.data());
    req->wait();

    // Verify data
    std::vector<int> h_recvbuf(total_recv);
    cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(int), cudaMemcpyDeviceToHost);

    auto displs = stComm::Utils::calculateDisplacements(recvcounts);
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < recvcounts[r]; ++i) {
            EXPECT_EQ(h_recvbuf[displs[r] + i], r * 100);
        }
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}

// Test alltoallv with auto displacement
TEST_F(NCCLCommTest, Alltoallv) {
    // Each rank sends 5 items to each other rank
    std::vector<int> sendcounts(size, 5);
    std::vector<int> recvcounts(size, 5);

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    // Allocate device memory
    int *d_sendbuf, *d_recvbuf;
    cudaMalloc(&d_sendbuf, total_send * sizeof(int));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(int));

    // Initialize send buffer
    std::vector<int> h_sendbuf(total_send);
    auto send_displs = stComm::Utils::calculateDisplacements(sendcounts);
    for (int dest = 0; dest < size; ++dest) {
        for (int i = 0; i < sendcounts[dest]; ++i) {
            h_sendbuf[send_displs[dest] + i] = rank * 1000 + dest * 100 + i;
        }
    }
    cudaMemcpy(d_sendbuf, h_sendbuf.data(), total_send * sizeof(int), cudaMemcpyHostToDevice);

    // Call alltoallv (displacement auto-calculated, uses internal stream)
    auto req = nccl_comm->alltoallv(d_sendbuf, sendcounts.data(),
                                    d_recvbuf, recvcounts.data());
    req->wait();

    // Verify received data
    std::vector<int> h_recvbuf(total_recv);
    cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(int), cudaMemcpyDeviceToHost);

    auto recv_displs = stComm::Utils::calculateDisplacements(recvcounts);
    for (int src = 0; src < size; ++src) {
        for (int i = 0; i < recvcounts[src]; ++i) {
            int expected = src * 1000 + rank * 100 + i;
            EXPECT_EQ(h_recvbuf[recv_displs[src] + i], expected);
        }
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}

// Test async operations
TEST_F(NCCLCommTest, AsyncOperations) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 100;
    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));

    if (rank == 0) {
        std::vector<int> h_data(N, rank);
        cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        auto req = nccl_comm->send(d_data, N, 1);

        // Check if operation is pending
        req->test();  // Test non-blocking check

        // Wait for completion
        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);
    } else if (rank == 1) {
        auto req = nccl_comm->recv(d_data, N, 0);

        // Wait for completion
        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

        std::vector<int> h_data(N);
        cudaMemcpy(h_data.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (auto val : h_data) {
            EXPECT_EQ(val, 0);
        }
    }

    cudaFree(d_data);
}
