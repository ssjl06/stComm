/**
 * @file test_nccl_comm.cpp
 * @brief Comprehensive tests for NCCLComm class with random data
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include <vector>
#include <random>
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

        // Initialize random generator
        rng.seed(54321 + rank);
    }

    void TearDown() override {
        nccl_comm.reset();
        mpi_comm.reset();
    }

    // Helper to generate random data
    template<typename T>
    void fillRandom(std::vector<T>& data) {
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(),
                                                   std::numeric_limits<T>::max());
            for (auto& val : data) {
                val = dist(rng);
            }
        } else {
            std::uniform_real_distribution<T> dist(-1000.0, 1000.0);
            for (auto& val : data) {
                val = dist(rng);
            }
        }
    }

    std::unique_ptr<stComm::MPIComm> mpi_comm;
    std::unique_ptr<stComm::NCCLComm> nccl_comm;
    int rank;
    int size;
    int device_id;
    std::mt19937 rng;
};

// ============================================================================
// Basic Interface Tests
// ============================================================================

TEST_F(NCCLCommTest, GetRank) {
    int r = nccl_comm->getRank();
    EXPECT_GE(r, 0);
    EXPECT_LT(r, size);
    EXPECT_EQ(r, rank);
}

TEST_F(NCCLCommTest, GetSize) {
    int s = nccl_comm->getSize();
    EXPECT_GT(s, 0);
    EXPECT_EQ(s, size);
}

TEST_F(NCCLCommTest, GetBackend) {
    EXPECT_EQ(nccl_comm->getBackend(), stComm::Backend::NCCL);
}

TEST_F(NCCLCommTest, GetHandle) {
    ncclComm_t handle = nccl_comm->getHandle();
    EXPECT_NE(handle, nullptr);
}

TEST_F(NCCLCommTest, Barrier) {
    // Test multiple barriers
    EXPECT_NO_THROW(nccl_comm->barrier());
    EXPECT_NO_THROW(nccl_comm->barrier());
    EXPECT_NO_THROW(nccl_comm->barrier());
}

TEST_F(NCCLCommTest, GroupStartEnd) {
    // Test group operations
    EXPECT_NO_THROW(nccl_comm->groupStart());
    EXPECT_NO_THROW(nccl_comm->groupEnd());
}

// ============================================================================
// Point-to-Point Communication Tests
// ============================================================================

TEST_F(NCCLCommTest, SendRecvFloat) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 10000;
    std::vector<float> h_send_data(N);
    std::vector<float> h_recv_data(N);

    float *d_send_data, *d_recv_data;
    cudaMalloc(&d_send_data, N * sizeof(float));
    cudaMalloc(&d_recv_data, N * sizeof(float));

    // Generate random data
    fillRandom(h_send_data);
    cudaMemcpy(d_send_data, h_send_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Ring communication
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    nccl_comm->groupStart();
    auto send_req = nccl_comm->send(d_send_data, N, send_to);
    auto recv_req = nccl_comm->recv(d_recv_data, N, recv_from);
    nccl_comm->groupEnd();

    EXPECT_NE(send_req, nullptr);
    EXPECT_NE(recv_req, nullptr);

    send_req->wait();
    recv_req->wait();

    EXPECT_EQ(send_req->getStatus(), stComm::Status::SUCCESS);
    EXPECT_EQ(recv_req->getStatus(), stComm::Status::SUCCESS);

    // Verify
    cudaMemcpy(h_recv_data.data(), d_recv_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Gather all send data to verify
    std::vector<float> all_send_data(N * size);
    MPI_Allgather(h_send_data.data(), N * sizeof(float), MPI_BYTE,
                  all_send_data.data(), N * sizeof(float), MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(h_recv_data[i], all_send_data[recv_from * N + i])
            << "Mismatch at index " << i;
    }

    cudaFree(d_send_data);
    cudaFree(d_recv_data);
}

TEST_F(NCCLCommTest, SendRecvInt) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 8000;
    std::vector<int> h_send_data(N);
    std::vector<int> h_recv_data(N);

    int *d_send_data, *d_recv_data;
    cudaMalloc(&d_send_data, N * sizeof(int));
    cudaMalloc(&d_recv_data, N * sizeof(int));

    fillRandom(h_send_data);
    cudaMemcpy(d_send_data, h_send_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    nccl_comm->groupStart();
    auto send_req = nccl_comm->send(d_send_data, N, send_to);
    auto recv_req = nccl_comm->recv(d_recv_data, N, recv_from);
    nccl_comm->groupEnd();

    send_req->wait();
    recv_req->wait();

    cudaMemcpy(h_recv_data.data(), d_recv_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> all_send_data(N * size);
    MPI_Allgather(h_send_data.data(), N * sizeof(int), MPI_BYTE,
                  all_send_data.data(), N * sizeof(int), MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(h_recv_data[i], all_send_data[recv_from * N + i]);
    }

    cudaFree(d_send_data);
    cudaFree(d_recv_data);
}

TEST_F(NCCLCommTest, SendRecvDouble) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 5000;
    std::vector<double> h_send_data(N);
    std::vector<double> h_recv_data(N);

    double *d_send_data, *d_recv_data;
    cudaMalloc(&d_send_data, N * sizeof(double));
    cudaMalloc(&d_recv_data, N * sizeof(double));

    fillRandom(h_send_data);
    cudaMemcpy(d_send_data, h_send_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    nccl_comm->groupStart();
    auto send_req = nccl_comm->send(d_send_data, N, send_to);
    auto recv_req = nccl_comm->recv(d_recv_data, N, recv_from);
    nccl_comm->groupEnd();

    send_req->wait();
    recv_req->wait();

    cudaMemcpy(h_recv_data.data(), d_recv_data, N * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> all_send_data(N * size);
    MPI_Allgather(h_send_data.data(), N * sizeof(double), MPI_BYTE,
                  all_send_data.data(), N * sizeof(double), MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(h_recv_data[i], all_send_data[recv_from * N + i]);
    }

    cudaFree(d_send_data);
    cudaFree(d_recv_data);
}

// ============================================================================
// Collective Communication Tests
// ============================================================================

TEST_F(NCCLCommTest, AllgathervUniformCounts) {
    const int sendcount = 1000;
    std::vector<int> h_sendbuf(sendcount);
    fillRandom(h_sendbuf);

    std::vector<int> recvcounts(size, sendcount);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    int *d_sendbuf, *d_recvbuf;
    cudaMalloc(&d_sendbuf, sendcount * sizeof(int));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(int));

    cudaMemcpy(d_sendbuf, h_sendbuf.data(), sendcount * sizeof(int), cudaMemcpyHostToDevice);

    auto req = nccl_comm->allgatherv(d_sendbuf, sendcount, d_recvbuf, recvcounts.data());
    EXPECT_NE(req, nullptr);
    req->wait();
    EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

    // Verify
    std::vector<int> h_recvbuf(total_recv);
    cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> expected(total_recv);
    MPI_Allgather(h_sendbuf.data(), sendcount * sizeof(int), MPI_BYTE,
                  expected.data(), sendcount * sizeof(int), MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(h_recvbuf[i], expected[i]) << "Mismatch at index " << i;
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}

TEST_F(NCCLCommTest, AllgathervVariableCounts) {
    int sendcount = (rank + 1) * 100;
    std::vector<int> h_sendbuf(sendcount);
    fillRandom(h_sendbuf);

    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (i + 1) * 100;
    }

    int total_recv = stComm::Utils::totalSize(recvcounts);

    int *d_sendbuf, *d_recvbuf;
    cudaMalloc(&d_sendbuf, sendcount * sizeof(int));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(int));

    cudaMemcpy(d_sendbuf, h_sendbuf.data(), sendcount * sizeof(int), cudaMemcpyHostToDevice);

    auto req = nccl_comm->allgatherv(d_sendbuf, sendcount, d_recvbuf, recvcounts.data());
    req->wait();

    // Verify
    std::vector<int> h_recvbuf(total_recv);
    cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(int), cudaMemcpyDeviceToHost);

    auto displs = stComm::Utils::calculateDisplacements(recvcounts);
    std::vector<int> byte_recvcounts(size);
    std::vector<int> byte_displs(size);
    for (int i = 0; i < size; ++i) {
        byte_recvcounts[i] = recvcounts[i] * sizeof(int);
        byte_displs[i] = displs[i] * sizeof(int);
    }

    std::vector<int> expected(total_recv);
    MPI_Allgatherv(h_sendbuf.data(), sendcount * sizeof(int), MPI_BYTE,
                   expected.data(), byte_recvcounts.data(), byte_displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(h_recvbuf[i], expected[i]) << "Variable allgatherv mismatch at index " << i;
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}

TEST_F(NCCLCommTest, AlltoallvUniformCounts) {
    const int count_per_rank = 50;
    std::vector<int> sendcounts(size, count_per_rank);
    std::vector<int> recvcounts(size, count_per_rank);

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    std::vector<int> h_sendbuf(total_send);
    fillRandom(h_sendbuf);

    int *d_sendbuf, *d_recvbuf;
    cudaMalloc(&d_sendbuf, total_send * sizeof(int));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(int));

    cudaMemcpy(d_sendbuf, h_sendbuf.data(), total_send * sizeof(int), cudaMemcpyHostToDevice);

    auto req = nccl_comm->alltoallv(d_sendbuf, sendcounts.data(),
                                    d_recvbuf, recvcounts.data());
    EXPECT_NE(req, nullptr);
    req->wait();
    EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

    // Verify
    std::vector<int> h_recvbuf(total_recv);
    cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> expected(total_recv);
    MPI_Alltoall(h_sendbuf.data(), count_per_rank * sizeof(int), MPI_BYTE,
                 expected.data(), count_per_rank * sizeof(int), MPI_BYTE,
                 MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(h_recvbuf[i], expected[i]) << "Alltoallv uniform mismatch at index " << i;
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}

TEST_F(NCCLCommTest, AlltoallvVariableCounts) {
    std::vector<int> sendcounts(size);
    std::vector<int> recvcounts(size);

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rank + 1) * 20;
        recvcounts[i] = (i + 1) * 20;
    }

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    std::vector<int> h_sendbuf(total_send);
    fillRandom(h_sendbuf);

    int *d_sendbuf, *d_recvbuf;
    cudaMalloc(&d_sendbuf, total_send * sizeof(int));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(int));

    cudaMemcpy(d_sendbuf, h_sendbuf.data(), total_send * sizeof(int), cudaMemcpyHostToDevice);

    auto req = nccl_comm->alltoallv(d_sendbuf, sendcounts.data(),
                                    d_recvbuf, recvcounts.data());
    req->wait();

    // Verify
    std::vector<int> h_recvbuf(total_recv);
    cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(int), cudaMemcpyDeviceToHost);

    auto sdispls = stComm::Utils::calculateDisplacements(sendcounts);
    auto rdispls = stComm::Utils::calculateDisplacements(recvcounts);

    std::vector<int> byte_sendcounts(size);
    std::vector<int> byte_sdispls(size);
    std::vector<int> byte_recvcounts(size);
    std::vector<int> byte_rdispls(size);

    for (int i = 0; i < size; ++i) {
        byte_sendcounts[i] = sendcounts[i] * sizeof(int);
        byte_sdispls[i] = sdispls[i] * sizeof(int);
        byte_recvcounts[i] = recvcounts[i] * sizeof(int);
        byte_rdispls[i] = rdispls[i] * sizeof(int);
    }

    std::vector<int> expected(total_recv);
    MPI_Alltoallv(h_sendbuf.data(), byte_sendcounts.data(), byte_sdispls.data(), MPI_BYTE,
                  expected.data(), byte_recvcounts.data(), byte_rdispls.data(), MPI_BYTE,
                  MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(h_recvbuf[i], expected[i]) << "Variable alltoallv mismatch at index " << i;
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}

// ============================================================================
// Async Operation Tests
// ============================================================================

TEST_F(NCCLCommTest, AsyncOperations) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 3000;
    std::vector<int> h_send_data(N);
    fillRandom(h_send_data);

    int *d_send_data, *d_recv_data;
    cudaMalloc(&d_send_data, N * sizeof(int));
    cudaMalloc(&d_recv_data, N * sizeof(int));

    cudaMemcpy(d_send_data, h_send_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    nccl_comm->groupStart();
    auto send_req = nccl_comm->send(d_send_data, N, send_to);
    auto recv_req = nccl_comm->recv(d_recv_data, N, recv_from);
    nccl_comm->groupEnd();

    // Test non-blocking check
    send_req->test();
    recv_req->test();

    send_req->wait();
    recv_req->wait();

    EXPECT_EQ(send_req->getStatus(), stComm::Status::SUCCESS);
    EXPECT_EQ(recv_req->getStatus(), stComm::Status::SUCCESS);

    cudaFree(d_send_data);
    cudaFree(d_recv_data);
}

// ============================================================================
// Mixed Type Tests
// ============================================================================

TEST_F(NCCLCommTest, AllgathervFloat) {
    const int sendcount = 500;
    std::vector<float> h_sendbuf(sendcount);
    fillRandom(h_sendbuf);

    std::vector<int> recvcounts(size, sendcount);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    float *d_sendbuf, *d_recvbuf;
    cudaMalloc(&d_sendbuf, sendcount * sizeof(float));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(float));

    cudaMemcpy(d_sendbuf, h_sendbuf.data(), sendcount * sizeof(float), cudaMemcpyHostToDevice);

    auto req = nccl_comm->allgatherv(d_sendbuf, sendcount, d_recvbuf, recvcounts.data());
    req->wait();

    std::vector<float> h_recvbuf(total_recv);
    cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> expected(total_recv);
    MPI_Allgather(h_sendbuf.data(), sendcount * sizeof(float), MPI_BYTE,
                  expected.data(), sendcount * sizeof(float), MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_FLOAT_EQ(h_recvbuf[i], expected[i]);
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}

TEST_F(NCCLCommTest, AlltoallvDouble) {
    const int count_per_rank = 30;
    std::vector<int> sendcounts(size, count_per_rank);
    std::vector<int> recvcounts(size, count_per_rank);

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    std::vector<double> h_sendbuf(total_send);
    fillRandom(h_sendbuf);

    double *d_sendbuf, *d_recvbuf;
    cudaMalloc(&d_sendbuf, total_send * sizeof(double));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(double));

    cudaMemcpy(d_sendbuf, h_sendbuf.data(), total_send * sizeof(double), cudaMemcpyHostToDevice);

    auto req = nccl_comm->alltoallv(d_sendbuf, sendcounts.data(),
                                    d_recvbuf, recvcounts.data());
    req->wait();

    std::vector<double> h_recvbuf(total_recv);
    cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> expected(total_recv);
    MPI_Alltoall(h_sendbuf.data(), count_per_rank * sizeof(double), MPI_BYTE,
                 expected.data(), count_per_rank * sizeof(double), MPI_BYTE,
                 MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_DOUBLE_EQ(h_recvbuf[i], expected[i]);
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}
