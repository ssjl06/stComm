/**
 * @file mpi_hello.cpp
 * @brief Basic MPI communication example using stComm
 *
 * This example demonstrates:
 * - MPIComm initialization and finalization
 * - Basic rank and size queries
 * - Simple send/recv communication
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // Initialize MPI
    stComm::MPIComm::initialize(&argc, &argv);

    // Create MPI communicator
    stComm::MPIComm comm;
    int rank = comm.getRank();
    int size = comm.getSize();

    std::cout << "Hello from rank " << rank << " of " << size << std::endl;

    // Barrier to synchronize all ranks
    comm.barrier();

    if (size >= 2) {
        // Simple send/recv example
        std::vector<int> data(10);

        if (rank == 0) {
            // Rank 0 sends data to rank 1
            for (int i = 0; i < 10; ++i) {
                data[i] = i * 10;
            }

            std::cout << "Rank 0: Sending data to rank 1" << std::endl;
            auto req = comm.send(data.data(), data.size(), 1, 0);
            req->wait();
            std::cout << "Rank 0: Data sent successfully" << std::endl;

        } else if (rank == 1) {
            // Rank 1 receives data from rank 0
            std::cout << "Rank 1: Receiving data from rank 0" << std::endl;
            auto req = comm.recv(data.data(), data.size(), 0, 0);
            req->wait();

            std::cout << "Rank 1: Received data: ";
            for (int val : data) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    // Finalize MPI
    stComm::MPIComm::finalize();
    return 0;
}
