/**
 * @file nccl_ring.cpp
 * @brief GPU ring communication via the stComm::Comm facade.
 *
 * Demonstrates:
 * - Device-enabled Comm via Comm::onDevice (NCCL bootstrap is internal — no
 *   manual ncclUniqueId handshake / ncclCommInitRank)
 * - Ring topology on GPU with NCCL send/recv + groupStart/groupEnd, reached
 *   through the comm.nccl() escape hatch (p2p + grouping are backend ops)
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    stComm::Comm::initialize(&argc, &argv);

    // Probe rank/size on a host-only Comm to pick the device and guard on GPUs.
    int rank = 0, size = 0;
    { stComm::Comm probe; rank = probe.getRank(); size = probe.getSize(); }

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (size > num_gpus) {
        if (rank == 0) {
            std::cerr << "Error: Need " << size << " GPUs but only "
                      << num_gpus << " available" << std::endl;
        }
        stComm::Comm::finalize();
        return 1;
    }
    if (size < 2) {
        if (rank == 0) {
            std::cerr << "This example requires at least 2 processes" << std::endl;
        }
        stComm::Comm::finalize();
        return 1;
    }

    // Each rank uses its own GPU. onDevice bootstraps NCCL on it internally.
    int device_id = rank;
    cudaSetDevice(device_id);
    stComm::Comm comm = stComm::Comm::onDevice(device_id);

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

    // p2p send/recv + grouping are NCCL-specific — reach them via comm.nccl().
    // IMPORTANT: group send/recv together to prevent deadlock.
    auto& nccl = comm.nccl();
    nccl.groupStart();
    auto send_req = nccl.send(d_send_data, N, next);
    auto recv_req = nccl.recv(d_recv_data, N, prev);
    nccl.groupEnd();

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

    stComm::Comm::finalize();
    return success ? 0 : 1;
}
