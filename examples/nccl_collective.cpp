/**
 * @file nccl_collective.cpp
 * @brief GPU collective communication using NCCL
 *
 * This example demonstrates:
 * - NCCL allgatherv with automatic displacement calculation
 * - NCCL alltoallv for all-to-all GPU communication
 * - Working with device memory
 * - Variable-sized collective operations on GPU
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

    // Each rank uses its own GPU
    int device_id = rank;
    cudaSetDevice(device_id);

    // Setup NCCL communicator
    ncclUniqueId nccl_id;
    if (rank == 0) {
        nccl_id = stComm::NCCLComm::getUniqueId();
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    stComm::NCCLComm nccl_comm;
    nccl_comm.initialize(rank, size, device_id, nccl_id);

    std::cout << "=== GPU " << device_id << " (Rank " << rank
              << ") starting collective operations ===" << std::endl;

    // ========================================================================
    // Example 1: allgatherv with variable sizes
    // ========================================================================
    {
        std::cout << "\nGPU " << device_id << ": Testing allgatherv..." << std::endl;

        // Each GPU sends different amount: (rank+1)*100 elements
        int sendcount = (rank + 1) * 100;

        std::vector<int> recvcounts(size);
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = (i + 1) * 100;
        }

        int total_recv = stComm::Utils::totalSize(recvcounts);

        // Allocate GPU memory
        int *d_sendbuf, *d_recvbuf;
        cudaMalloc(&d_sendbuf, sendcount * sizeof(int));
        cudaMalloc(&d_recvbuf, total_recv * sizeof(int));

        // Initialize send buffer on host
        std::vector<int> h_sendbuf(sendcount);
        for (int i = 0; i < sendcount; ++i) {
            h_sendbuf[i] = rank * 10000 + i;
        }
        cudaMemcpy(d_sendbuf, h_sendbuf.data(), sendcount * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Perform allgatherv (displacement automatically calculated)
        auto req = nccl_comm.allgatherv(d_sendbuf, sendcount,
                                        d_recvbuf, recvcounts.data());
        req->wait();

        // Verify results
        std::vector<int> h_recvbuf(total_recv);
        cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(int),
                   cudaMemcpyDeviceToHost);

        auto displs = stComm::Utils::calculateDisplacements(recvcounts);
        bool success = true;
        for (int r = 0; r < size; ++r) {
            for (int i = 0; i < recvcounts[r]; ++i) {
                int expected = r * 10000 + i;
                if (h_recvbuf[displs[r] + i] != expected) {
                    success = false;
                    std::cerr << "GPU " << device_id << ": allgatherv verification failed!"
                              << std::endl;
                    break;
                }
            }
            if (!success) break;
        }

        if (success) {
            std::cout << "GPU " << device_id << ": allgatherv successful! "
                      << "Gathered " << total_recv << " total elements" << std::endl;
        }

        cudaFree(d_sendbuf);
        cudaFree(d_recvbuf);
    }

    // ========================================================================
    // Example 2: alltoallv - Personalized all-to-all on GPU
    // ========================================================================
    {
        std::cout << "\nGPU " << device_id << ": Testing alltoallv..." << std::endl;

        // Each GPU sends 50 elements to every other GPU
        std::vector<int> sendcounts(size, 50);
        std::vector<int> recvcounts(size, 50);

        int total_send = stComm::Utils::totalSize(sendcounts);
        int total_recv = stComm::Utils::totalSize(recvcounts);

        // Allocate GPU memory
        int *d_sendbuf, *d_recvbuf;
        cudaMalloc(&d_sendbuf, total_send * sizeof(int));
        cudaMalloc(&d_recvbuf, total_recv * sizeof(int));

        // Initialize send buffer
        std::vector<int> h_sendbuf(total_send);
        auto send_displs = stComm::Utils::calculateDisplacements(sendcounts);
        for (int dest = 0; dest < size; ++dest) {
            for (int i = 0; i < sendcounts[dest]; ++i) {
                h_sendbuf[send_displs[dest] + i] = rank * 100000 + dest * 1000 + i;
            }
        }
        cudaMemcpy(d_sendbuf, h_sendbuf.data(), total_send * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Perform alltoallv (displacement automatically calculated)
        auto req = nccl_comm.alltoallv(d_sendbuf, sendcounts.data(),
                                       d_recvbuf, recvcounts.data());
        req->wait();

        // Verify results
        std::vector<int> h_recvbuf(total_recv);
        cudaMemcpy(h_recvbuf.data(), d_recvbuf, total_recv * sizeof(int),
                   cudaMemcpyDeviceToHost);

        auto recv_displs = stComm::Utils::calculateDisplacements(recvcounts);
        bool success = true;
        for (int src = 0; src < size; ++src) {
            for (int i = 0; i < recvcounts[src]; ++i) {
                int expected = src * 100000 + rank * 1000 + i;
                if (h_recvbuf[recv_displs[src] + i] != expected) {
                    success = false;
                    std::cerr << "GPU " << device_id << ": alltoallv verification failed!"
                              << std::endl;
                    break;
                }
            }
            if (!success) break;
        }

        if (success) {
            std::cout << "GPU " << device_id << ": alltoallv successful! "
                      << "Exchanged " << total_send << " elements" << std::endl;
        }

        cudaFree(d_sendbuf);
        cudaFree(d_recvbuf);
    }

    std::cout << "\nGPU " << device_id << ": All collective operations completed successfully!"
              << std::endl;

    stComm::MPIComm::finalize();
    return 0;
}
