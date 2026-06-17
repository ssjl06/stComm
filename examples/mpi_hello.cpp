/**
 * @file mpi_hello.cpp
 * @brief Basic host communication via the unified stComm::Comm facade.
 *
 * Demonstrates:
 * - Comm lifecycle (initialize/finalize) and a host-only Comm
 * - rank/size queries and barrier through the facade
 * - point-to-point send/recv reached through the comm.mpi() escape hatch
 *   (the facade exposes collectives; p2p lives on the MPI backend)
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    stComm::Comm::initialize(&argc, &argv);

    stComm::Comm comm;                 // host-only Comm
    int rank = comm.getRank();
    int size = comm.getSize();

    std::cout << "Hello from rank " << rank << " of " << size << std::endl;

    // Barrier to synchronize all ranks
    comm.barrier();

    if (size >= 2) {
        std::vector<int> data(10);

        if (rank == 0) {
            for (int i = 0; i < 10; ++i) {
                data[i] = i * 10;
            }

            std::cout << "Rank 0: Sending data to rank 1" << std::endl;
            // send/recv are backend p2p ops — reach them via the MPI escape hatch.
            comm.mpi().send(data.data(), data.size(), 1, 0)->wait();
            std::cout << "Rank 0: Data sent successfully" << std::endl;

        } else if (rank == 1) {
            std::cout << "Rank 1: Receiving data from rank 0" << std::endl;
            comm.mpi().recv(data.data(), data.size(), 0, 0)->wait();

            std::cout << "Rank 1: Received data: ";
            for (int val : data) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    stComm::Comm::finalize();
    return 0;
}
