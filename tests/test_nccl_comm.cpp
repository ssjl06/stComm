#include <gtest/gtest.h>
#include "stComm/nccl_comm.h"
#include "stComm/mpi_comm.h"
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

class NCCLCommTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Initialize MPI for NCCL coordination
        mpi_comm = std::make_unique<stComm::MPIComm>();
        rank = mpi_comm->getRank();
        size = mpi_comm->getSize();

        // Use device based on rank (modulo device count)
        device_id = rank % device_count;

        // Get NCCL unique ID from rank 0
        ncclUniqueId nccl_id;
        if (rank == 0) {
            nccl_id = stComm::NCCLComm::getUniqueId();
        }

        // Broadcast NCCL ID to all ranks
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Initialize NCCL communicator
        nccl_comm = std::make_unique<stComm::NCCLComm>();
        nccl_comm->initialize(rank, size, device_id, nccl_id);

        // Set CUDA device
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

TEST_F(NCCLCommTest, BasicInfo) {
    EXPECT_EQ(nccl_comm->getRank(), rank);
    EXPECT_EQ(nccl_comm->getSize(), size);
}

TEST_F(NCCLCommTest, SendRecvFloat) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int count = 100;
    std::vector<float> host_send_data(count);
    std::vector<float> host_recv_data(count, 0.0f);

    // Initialize send data
    for (int i = 0; i < count; ++i) {
        host_send_data[i] = rank * 100.0f + i;
    }

    // Allocate device memory
    float *dev_send_data, *dev_recv_data;
    cudaMalloc(&dev_send_data, count * sizeof(float));
    cudaMalloc(&dev_recv_data, count * sizeof(float));

    // Copy to device
    cudaMemcpy(dev_send_data, host_send_data.data(), count * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(dev_recv_data, 0, count * sizeof(float));

    stComm::RequestPtr req;

    if (rank == 0) {
        req = nccl_comm->send(dev_send_data, count, 1);
    } else if (rank == 1) {
        req = nccl_comm->recv(dev_recv_data, count, 0);
    }

    if (req) {
        EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);
    }

    // Copy back to host and verify
    if (rank == 1) {
        cudaMemcpy(host_recv_data.data(), dev_recv_data, count * sizeof(float),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < count; ++i) {
            EXPECT_FLOAT_EQ(host_recv_data[i], static_cast<float>(i));
        }
    }

    cudaFree(dev_send_data);
    cudaFree(dev_recv_data);
}

TEST_F(NCCLCommTest, AllGatherInt) {
    const int count = 50;
    std::vector<int> host_send_data(count);
    std::vector<int> host_recv_data(count * size, 0);

    // Initialize send data
    std::iota(host_send_data.begin(), host_send_data.end(), rank * 1000);

    // Allocate device memory
    int *dev_send_data, *dev_recv_data;
    cudaMalloc(&dev_send_data, count * sizeof(int));
    cudaMalloc(&dev_recv_data, count * size * sizeof(int));

    // Copy to device
    cudaMemcpy(dev_send_data, host_send_data.data(), count * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemset(dev_recv_data, 0, count * size * sizeof(int));

    // Perform allgather
    auto req = nccl_comm->allgather(dev_send_data, dev_recv_data, count);

    ASSERT_NE(req, nullptr);
    EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);

    // Copy back and verify
    cudaMemcpy(host_recv_data.data(), dev_recv_data, count * size * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < count; ++i) {
            EXPECT_EQ(host_recv_data[r * count + i], r * 1000 + i);
        }
    }

    cudaFree(dev_send_data);
    cudaFree(dev_recv_data);
}

TEST_F(NCCLCommTest, AllReduceSum) {
    const int count = 100;
    std::vector<float> host_send_data(count);
    std::vector<float> host_recv_data(count, 0.0f);

    // Initialize send data - each rank contributes rank value
    for (int i = 0; i < count; ++i) {
        host_send_data[i] = static_cast<float>(rank);
    }

    // Allocate device memory
    float *dev_send_data, *dev_recv_data;
    cudaMalloc(&dev_send_data, count * sizeof(float));
    cudaMalloc(&dev_recv_data, count * sizeof(float));

    // Copy to device
    cudaMemcpy(dev_send_data, host_send_data.data(), count * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(dev_recv_data, 0, count * sizeof(float));

    // Perform allreduce
    auto req = nccl_comm->allreduce(dev_send_data, dev_recv_data, count, ncclSum);

    ASSERT_NE(req, nullptr);
    EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);

    // Copy back and verify
    cudaMemcpy(host_recv_data.data(), dev_recv_data, count * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Sum should be 0 + 1 + 2 + ... + (size-1) = size*(size-1)/2
    float expected_sum = static_cast<float>(size * (size - 1) / 2);

    for (int i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(host_recv_data[i], expected_sum);
    }

    cudaFree(dev_send_data);
    cudaFree(dev_recv_data);
}

TEST_F(NCCLCommTest, AsyncTest) {
    const int count = 50;
    std::vector<double> host_data(count);

    std::iota(host_data.begin(), host_data.end(), rank * 100.0);

    double *dev_send_data, *dev_recv_data;
    cudaMalloc(&dev_send_data, count * sizeof(double));
    cudaMalloc(&dev_recv_data, count * size * sizeof(double));

    cudaMemcpy(dev_send_data, host_data.data(), count * sizeof(double),
               cudaMemcpyHostToDevice);

    // Create custom stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Perform async allgather with custom stream
    auto req = nccl_comm->allgather(dev_send_data, dev_recv_data, count, stream);

    ASSERT_NE(req, nullptr);

    // Test non-blocking check
    bool completed = false;
    for (int i = 0; i < 10; ++i) {
        req->test(completed);
        if (completed) {
            break;
        }
    }

    // Wait for completion
    EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);

    cudaStreamDestroy(stream);
    cudaFree(dev_send_data);
    cudaFree(dev_recv_data);
}
