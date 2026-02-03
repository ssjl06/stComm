/**
 * @file nccl_ring.cpp
 * @brief GPU ring communication using NCCL
 *
 * This example demonstrates:
 * - NCCL communicator initialization
 * - Ring topology on GPU with send/recv
 * - Proper use of groupStart/groupEnd to prevent deadlock
 * - GPU memory management
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    // Initialize MPI for coordination
    stComm::MPIComm::initialize(&argc, &argv);

    stComm::MPIComm mpi_comm;
    int rank = mpi_comm.getRank();
    int size = mpi_comm.getSize();

    // Check GPU availability
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);

    if (size > num_gpus) {
        if (rank == 0) {
            std::cerr << "Error: Need " << size << " GPUs but only "
                      << num_gpus << " available" << std::endl;
        }
        stComm::MPIComm::finalize();
        return 1;
    }

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "This example requires at least 2 processes" << std::endl;
        }
        stComm::MPIComm::finalize();
        return 1;
    }

    // Each rank uses its own GPU
    int device_id = rank;
    cudaSetDevice(device_id);

    // Get NCCL unique ID (broadcast from rank 0)
    ncclUniqueId nccl_id;
    if (rank == 0) {
        nccl_id = stComm::NCCLComm::getUniqueId();
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    stComm::NCCLComm nccl_comm;
    nccl_comm.initialize(rank, size, device_id, nccl_id);

    std::cout << "Rank " << rank << ": Using GPU " << device_id << std::endl;

    // Ring topology
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    const int N = 1000;

    // Allocate GPU memory
    float *d_send_data, *d_recv_data;
    cudaMalloc(&d_send_data, N * sizeof(float));
    cudaMalloc(&d_recv_data, N * sizeof(float));

    // Initialize send data on host
    std::vector<float> h_send_data(N);
    for (int i = 0; i < N; ++i) {
        h_send_data[i] = rank * 1000.0f + i;
    }

    // Copy to GPU
    cudaMemcpy(d_send_data, h_send_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Rank " << rank << ": Sending to GPU " << next
              << ", receiving from GPU " << prev << std::endl;

    // IMPORTANT: Group send/recv together to prevent deadlock
    nccl_comm.groupStart();
    auto send_req = nccl_comm.send(d_send_data, N, next);
    auto recv_req = nccl_comm.recv(d_recv_data, N, prev);
    nccl_comm.groupEnd();

    // Wait for completion
    send_req->wait();
    recv_req->wait();

    // Copy result back to host for verification
    std::vector<float> h_recv_data(N);
    cudaMemcpy(h_recv_data.data(), d_recv_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = prev * 1000.0f + i;
        if (h_recv_data[i] != expected) {
            success = false;
            std::cerr << "Rank " << rank << ": Data verification failed at index " << i
                      << " (expected " << expected << ", got " << h_recv_data[i] << ")"
                      << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Rank " << rank << ": GPU ring communication successful! "
                  << "Received correct data from GPU " << prev << std::endl;
    }

    // Cleanup
    cudaFree(d_send_data);
    cudaFree(d_recv_data);

    stComm::MPIComm::finalize();
    return success ? 0 : 1;
}
