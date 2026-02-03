/**
 * @file mpi_collective.cpp
 * @brief Collective communication with automatic displacement calculation
 *
 * This example demonstrates:
 * - allgatherv: Gather variable-sized data from all ranks
 * - alltoallv: All-to-all personalized exchange with variable sizes
 * - Automatic displacement calculation (no manual offset computation needed!)
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);

    stComm::MPIComm comm;
    int rank = comm.getRank();
    int size = comm.getSize();

    std::cout << "=== Rank " << rank << " starting collective communication example ===" << std::endl;

    // ========================================================================
    // Example 1: allgatherv - Each rank sends different amount of data
    // ========================================================================
    {
        std::cout << "\nRank " << rank << ": Testing allgatherv..." << std::endl;

        // Each rank sends (rank+1)*10 elements
        int sendcount = (rank + 1) * 10;
        std::vector<int> sendbuf(sendcount);
        for (int i = 0; i < sendcount; ++i) {
            sendbuf[i] = rank * 100 + i;
        }

        // Prepare receive counts (each rank will receive different amounts)
        std::vector<int> recvcounts(size);
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = (i + 1) * 10;
        }

        // Calculate total receive size
        int total_recv = stComm::Utils::totalSize(recvcounts);
        std::vector<int> recvbuf(total_recv);

        // Call allgatherv - displacement is automatically calculated!
        auto req = comm.allgatherv(sendbuf.data(), sendcount,
                                   recvbuf.data(), recvcounts.data());
        req->wait();

        // Verify results
        auto displs = stComm::Utils::calculateDisplacements(recvcounts);
        bool success = true;
        for (int r = 0; r < size; ++r) {
            for (int i = 0; i < recvcounts[r]; ++i) {
                int expected = r * 100 + i;
                if (recvbuf[displs[r] + i] != expected) {
                    success = false;
                    std::cerr << "Rank " << rank << ": allgatherv verification failed!" << std::endl;
                    break;
                }
            }
        }

        if (success) {
            std::cout << "Rank " << rank << ": allgatherv successful! "
                      << "Received " << total_recv << " total elements" << std::endl;
        }
    }

    // ========================================================================
    // Example 2: alltoallv - Personalized all-to-all exchange
    // ========================================================================
    {
        std::cout << "\nRank " << rank << ": Testing alltoallv..." << std::endl;

        // Each rank sends different amounts to each destination
        std::vector<int> sendcounts(size);
        std::vector<int> recvcounts(size);

        // Rank i sends (i+1) elements to rank j
        // Rank i receives (j+1) elements from rank j
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = rank + 1;
            recvcounts[i] = i + 1;
        }

        int total_send = stComm::Utils::totalSize(sendcounts);
        int total_recv = stComm::Utils::totalSize(recvcounts);

        std::vector<int> sendbuf(total_send);
        std::vector<int> recvbuf(total_recv);

        // Fill send buffer with data for each destination
        auto send_displs = stComm::Utils::calculateDisplacements(sendcounts);
        for (int dest = 0; dest < size; ++dest) {
            for (int i = 0; i < sendcounts[dest]; ++i) {
                sendbuf[send_displs[dest] + i] = rank * 1000 + dest * 100 + i;
            }
        }

        // Call alltoallv - displacement is automatically calculated!
        auto req = comm.alltoallv(sendbuf.data(), sendcounts.data(),
                                  recvbuf.data(), recvcounts.data());
        req->wait();

        // Verify results
        auto recv_displs = stComm::Utils::calculateDisplacements(recvcounts);
        bool success = true;
        for (int src = 0; src < size; ++src) {
            for (int i = 0; i < recvcounts[src]; ++i) {
                int expected = src * 1000 + rank * 100 + i;
                if (recvbuf[recv_displs[src] + i] != expected) {
                    success = false;
                    std::cerr << "Rank " << rank << ": alltoallv verification failed!" << std::endl;
                    break;
                }
            }
        }

        if (success) {
            std::cout << "Rank " << rank << ": alltoallv successful! "
                      << "Sent " << total_send << " elements, received " << total_recv << " elements"
                      << std::endl;
        }
    }

    std::cout << "\nRank " << rank << ": All collective operations completed successfully!" << std::endl;

    stComm::MPIComm::finalize();
    return 0;
}
