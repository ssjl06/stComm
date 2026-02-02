/**
 * @file unified_interface.cpp
 * @brief Example showing unified interface for MPI and NCCL
 *
 * This demonstrates how to use the same API for both CPU (MPI) and GPU (NCCL)
 * communication through the unified Communicator class.
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>
#include <iomanip>

void mpi_example() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "MPI Backend Example (CPU)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create communicator with MPI backend
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    std::cout << "[MPI] Rank " << rank << " / " << size << std::endl;

    // Example 1: Simple send/recv
    if (size >= 2) {
        std::vector<float> data(100);

        if (rank == 0) {
            for (int i = 0; i < 100; ++i) {
                data[i] = i * 1.5f;
            }
            auto req = comm.send(data.data(), data.size(), 1);
            req->wait();
            std::cout << "[MPI Rank 0] Sent data to rank 1" << std::endl;
        } else if (rank == 1) {
            auto req = comm.recv(data.data(), data.size(), 0);
            req->wait();
            std::cout << "[MPI Rank 1] Received data, first element = " << data[0] << std::endl;
        }
    }

    comm.barrier();

    // Example 2: allgather_vec (high-level API)
    {
        std::vector<int> my_data(3);
        for (int i = 0; i < 3; ++i) {
            my_data[i] = rank * 10 + i;
        }

        auto result = comm.allgather_vec(my_data);

        if (rank == 0) {
            std::cout << "[MPI] Allgather result: ";
            for (size_t i = 0; i < result.size(); ++i) {
                std::cout << result[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    comm.barrier();

    // Example 3: allgatherv_auto (variable counts with auto displacement)
    {
        int my_count = rank + 1;
        std::vector<int> sendbuf(my_count);
        for (int i = 0; i < my_count; ++i) {
            sendbuf[i] = rank * 100 + i;
        }

        std::vector<int> recvcounts(size);
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = i + 1;
        }

        int total = stComm::Utils::total_size(recvcounts);
        std::vector<int> recvbuf(total);

        // Auto displacement - no manual calculation needed!
        auto req = comm.allgatherv_auto(sendbuf.data(), my_count,
                                        recvbuf.data(), recvcounts.data());
        req->wait();

        if (rank == 0) {
            std::cout << "[MPI] Allgatherv (auto) result: ";
            for (int val : recvbuf) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    comm.barrier();

    // Example 4: allreduce
    {
        std::vector<int> sendbuf(5, rank + 1);
        std::vector<int> recvbuf(5);

        auto req = comm.allreduce(sendbuf.data(), recvbuf.data(), 5);
        req->wait();

        if (rank == 0) {
            int expected_sum = size * (size + 1) / 2;
            std::cout << "[MPI] Allreduce sum: " << recvbuf[0]
                      << " (expected: " << expected_sum << ")" << std::endl;
        }
    }
}

void nccl_example() {
#ifdef __CUDA_RUNTIME_H__
    std::cout << "\n========================================" << std::endl;
    std::cout << "NCCL Backend Example (GPU)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // First, initialize MPI for coordination
    stComm::Communicator mpi_comm(stComm::Backend::MPI);
    int rank = mpi_comm.getRank();
    int size = mpi_comm.getSize();
    int device_id = rank % 4;  // Assume 4 GPUs per node

    // Get NCCL unique ID (rank 0 creates it)
    ncclUniqueId nccl_id;
    if (rank == 0) {
        nccl_id = stComm::NCCLComm::getUniqueId();
    }
    // Broadcast to all ranks using MPI
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Create communicator with NCCL backend
    stComm::Communicator comm(stComm::Backend::NCCL, rank, size, device_id, nccl_id);

    std::cout << "[NCCL] Rank " << rank << " using GPU " << device_id << std::endl;

    // Example 1: Allreduce on GPU
    {
        const int N = 1000;
        float *d_data, *d_result;
        cudaMalloc(&d_data, N * sizeof(float));
        cudaMalloc(&d_result, N * sizeof(float));

        // Initialize with rank value
        std::vector<float> h_data(N, rank * 1.0f);
        cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // AllReduce on GPU
        auto req = comm.allreduce(d_data, d_result, N);
        req->wait();

        // Copy result back
        std::vector<float> h_result(N);
        cudaMemcpy(h_result.data(), d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

        if (rank == 0) {
            float expected = (size - 1) * size / 2.0f;
            std::cout << "[NCCL] Allreduce result: " << h_result[0]
                      << " (expected: " << expected << ")" << std::endl;
        }

        cudaFree(d_data);
        cudaFree(d_result);
    }

    // Example 2: Allgather on GPU
    {
        const int count = 100;
        float *d_sendbuf, *d_recvbuf;
        cudaMalloc(&d_sendbuf, count * sizeof(float));
        cudaMalloc(&d_recvbuf, count * size * sizeof(float));

        // Initialize with rank-specific data
        std::vector<float> h_sendbuf(count, rank * 10.0f);
        cudaMemcpy(d_sendbuf, h_sendbuf.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Allgather on GPU
        auto req = comm.allgather(d_sendbuf, count, d_recvbuf);
        req->wait();

        // Verify result
        std::vector<float> h_recvbuf(count * size);
        cudaMemcpy(h_recvbuf.data(), d_recvbuf, count * size * sizeof(float), cudaMemcpyDeviceToHost);

        if (rank == 0) {
            std::cout << "[NCCL] Allgather first elements from each rank: ";
            for (int i = 0; i < size; ++i) {
                std::cout << h_recvbuf[i * count] << " ";
            }
            std::cout << std::endl;
        }

        cudaFree(d_sendbuf);
        cudaFree(d_recvbuf);
    }

    // Example 3: Allgatherv with auto displacement
    {
        std::vector<int> counts(size);
        for (int i = 0; i < size; ++i) {
            counts[i] = (i + 1) * 10;
        }

        int my_count = (rank + 1) * 10;
        int total = stComm::Utils::total_size(counts);

        int *d_sendbuf, *d_recvbuf;
        cudaMalloc(&d_sendbuf, my_count * sizeof(int));
        cudaMalloc(&d_recvbuf, total * sizeof(int));

        // Initialize
        std::vector<int> h_sendbuf(my_count, rank);
        cudaMemcpy(d_sendbuf, h_sendbuf.data(), my_count * sizeof(int), cudaMemcpyHostToDevice);

        // Allgatherv with auto displacement!
        auto req = comm.allgatherv_auto(d_sendbuf, my_count, d_recvbuf, counts.data());
        req->wait();

        // Verify
        std::vector<int> h_recvbuf(total);
        cudaMemcpy(h_recvbuf.data(), d_recvbuf, total * sizeof(int), cudaMemcpyDeviceToHost);

        if (rank == 0) {
            std::cout << "[NCCL] Allgatherv (auto) received " << total << " elements" << std::endl;
        }

        cudaFree(d_sendbuf);
        cudaFree(d_recvbuf);
    }

    std::cout << "[NCCL] Rank " << rank << " completed all operations" << std::endl;
#else
    std::cout << "NCCL example skipped (CUDA not available)" << std::endl;
#endif
}

int main(int argc, char** argv) {
    // Initialize MPI
    stComm::MPIComm::initialize(&argc, &argv);

    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║  stComm Unified Interface Example     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝" << std::endl;

    // Run MPI example
    mpi_example();

    // Run NCCL example if CUDA is available
    stComm::Communicator temp(stComm::Backend::MPI);
    if (temp.getRank() == 0) {
        std::cout << "\nNote: NCCL example requires CUDA-enabled build" << std::endl;
    }

    // Uncomment to run NCCL example (requires CUDA)
    // nccl_example();

    // Finalize
    stComm::MPIComm::finalize();

    if (temp.getRank() == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "All examples completed!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "\nKey Benefit: Same API works for both MPI and NCCL!" << std::endl;
        std::cout << "- Change Backend::MPI to Backend::NCCL" << std::endl;
        std::cout << "- Switch host memory to device memory" << std::endl;
        std::cout << "- Everything else stays the same!" << std::endl;
    }

    return 0;
}
