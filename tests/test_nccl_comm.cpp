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

        // Get number of available GPUs
        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);

        // Skip test if not enough GPUs for all ranks
        if (size > num_gpus) {
            GTEST_SKIP() << "Test requires " << size << " GPUs but only "
                         << num_gpus << " available. Each rank needs a unique GPU.";
        }

        device_id = rank;  // Each rank uses its own GPU

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

// Test send/recv with ring pattern (all ranks participate)
TEST_F(NCCLCommTest, SendRecv) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 100;
    float *d_send_data, *d_recv_data;
    cudaMalloc(&d_send_data, N * sizeof(float));
    cudaMalloc(&d_recv_data, N * sizeof(float));

    // Prepare send data
    std::vector<float> h_send_data(N);
    for (int i = 0; i < N; ++i) {
        h_send_data[i] = rank * 1000 + i * 1.5f;
    }
    cudaMemcpy(d_send_data, h_send_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Ring communication: rank i sends to (i+1)%size, receives from (i-1+size)%size
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    // Group send and recv together to prevent deadlock
    nccl_comm->groupStart();
    auto send_req = nccl_comm->send(d_send_data, N, send_to);
    auto recv_req = nccl_comm->recv(d_recv_data, N, recv_from);
    nccl_comm->groupEnd();

    send_req->wait();
    recv_req->wait();

    // Verify received data
    std::vector<float> h_recv_data(N);
    cudaMemcpy(h_recv_data.data(), d_recv_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        float expected = recv_from * 1000 + i * 1.5f;
        EXPECT_NEAR(h_recv_data[i], expected, 1e-6f);
    }

    cudaFree(d_send_data);
    cudaFree(d_recv_data);
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

// Test async operations with ring pattern (all ranks participate)
TEST_F(NCCLCommTest, AsyncOperations) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 100;
    int *d_send_data, *d_recv_data;
    cudaMalloc(&d_send_data, N * sizeof(int));
    cudaMalloc(&d_recv_data, N * sizeof(int));

    // Prepare send data
    std::vector<int> h_send_data(N, rank);
    cudaMemcpy(d_send_data, h_send_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Ring communication
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    // Group send and recv together to prevent deadlock
    nccl_comm->groupStart();
    auto send_req = nccl_comm->send(d_send_data, N, send_to);
    auto recv_req = nccl_comm->recv(d_recv_data, N, recv_from);
    nccl_comm->groupEnd();

    // Test non-blocking check
    send_req->test();
    recv_req->test();

    // Wait for completion
    send_req->wait();
    recv_req->wait();
    EXPECT_EQ(send_req->getStatus(), stComm::Status::SUCCESS);
    EXPECT_EQ(recv_req->getStatus(), stComm::Status::SUCCESS);

    // Verify received data
    std::vector<int> h_recv_data(N);
    cudaMemcpy(h_recv_data.data(), d_recv_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (auto val : h_recv_data) {
        EXPECT_EQ(val, recv_from);
    }

    cudaFree(d_send_data);
    cudaFree(d_recv_data);
}
