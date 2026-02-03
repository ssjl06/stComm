/**
 * @file mpi_ring.cpp
 * @brief Ring communication pattern using stComm MPI
 *
 * This example demonstrates:
 * - Ring topology communication (each rank sends to next, receives from previous)
 * - Non-blocking send/recv operations
 * - Data passing around a ring
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);

    stComm::MPIComm comm;
    int rank = comm.getRank();
    int size = comm.getSize();

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "This example requires at least 2 processes" << std::endl;
        }
        stComm::MPIComm::finalize();
        return 1;
    }

    // Ring topology: rank i sends to (i+1)%size, receives from (i-1+size)%size
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    const int N = 100;
    std::vector<double> send_data(N);
    std::vector<double> recv_data(N);

    // Each rank prepares its own data
    for (int i = 0; i < N; ++i) {
        send_data[i] = rank * 1000.0 + i;
    }

    std::cout << "Rank " << rank << ": Sending to " << next
              << ", receiving from " << prev << std::endl;

    // Non-blocking send and receive
    auto send_req = comm.send(send_data.data(), N, next, 0);
    auto recv_req = comm.recv(recv_data.data(), N, prev, 0);

    // Wait for both operations to complete
    send_req->wait();
    recv_req->wait();

    // Verify received data
    bool success = true;
    for (int i = 0; i < N; ++i) {
        double expected = prev * 1000.0 + i;
        if (recv_data[i] != expected) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Rank " << rank << ": Ring communication successful! "
                  << "Received correct data from rank " << prev << std::endl;
    } else {
        std::cerr << "Rank " << rank << ": Data verification failed!" << std::endl;
    }

    stComm::MPIComm::finalize();
    return success ? 0 : 1;
}
