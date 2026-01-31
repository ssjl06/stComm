/**
 * @file simple_usage.cpp
 * @brief Simple examples showing the new high-level API
 *
 * This demonstrates how displacement is automatically calculated,
 * making the API much easier to use than raw MPI/NCCL.
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>
#include <iomanip>

void print_data(const std::vector<int>& data, const std::string& label, int rank) {
    std::cout << "[Rank " << rank << "] " << label << ": ";
    for (auto val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Initialize MPI
    stComm::MPIComm::initialize(&argc, &argv);
    stComm::MPIComm comm;

    int rank = comm.getRank();
    int size = comm.getSize();

    std::cout << "========================================" << std::endl;
    std::cout << "stComm High-Level API Examples" << std::endl;
    std::cout << "Rank: " << rank << " / " << size << std::endl;
    std::cout << "========================================\n" << std::endl;

    // ========================================
    // Example 1: allgather_vec (simplest!)
    // ========================================
    {
        std::cout << "[Example 1] allgather_vec - All ranks send same count\n" << std::endl;

        // Each rank has some data to share
        std::vector<int> my_data(3);
        for (int i = 0; i < 3; ++i) {
            my_data[i] = rank * 10 + i;
        }

        print_data(my_data, "My data", rank);

        // OLD WAY: You would need to allocate recvbuf, calculate sizes, etc.
        // NEW WAY: Just call allgather_vec!
        auto gathered = comm.allgather_vec(my_data);

        comm.barrier();
        if (rank == 0) {
            std::cout << "\nGathered data from all ranks: ";
            for (auto val : gathered) {
                std::cout << val << " ";
            }
            std::cout << "\n" << std::endl;
        }
    }

    comm.barrier();

    // ========================================
    // Example 2: allgatherv_auto - Different counts, auto displacement!
    // ========================================
    {
        std::cout << "\n[Example 2] allgatherv_auto - Different counts per rank\n" << std::endl;

        // Each rank sends different amount of data
        int my_count = rank + 1;  // Rank 0 sends 1, rank 1 sends 2, etc.
        std::vector<int> my_data(my_count);
        for (int i = 0; i < my_count; ++i) {
            my_data[i] = rank * 100 + i;
        }

        print_data(my_data, "My data", rank);

        // Define how much each rank will send
        std::vector<int> recvcounts(size);
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = i + 1;
        }

        // Calculate total receive size
        int total_recv = stComm::Utils::total_size(recvcounts);
        std::vector<int> recvbuf(total_recv);

        // OLD WAY: You would need to manually calculate displacements!
        // int displs[size];
        // displs[0] = 0;
        // for (int i = 1; i < size; ++i) {
        //     displs[i] = displs[i-1] + recvcounts[i-1];
        // }
        // auto req = comm.allgatherv(my_data.data(), my_count, recvbuf.data(), recvcounts.data(), displs);

        // NEW WAY: Displacements calculated automatically!
        auto req = comm.allgatherv_auto(my_data.data(), my_count, recvbuf.data(), recvcounts.data());
        req->wait();

        comm.barrier();
        if (rank == 0) {
            std::cout << "\nGathered data (variable lengths): ";
            for (auto val : recvbuf) {
                std::cout << val << " ";
            }
            std::cout << "\n" << std::endl;
        }
    }

    comm.barrier();

    // ========================================
    // Example 3: allgatherv_vec - Even simpler with vectors!
    // ========================================
    {
        std::cout << "\n[Example 3] allgatherv_vec - Vector-based API\n" << std::endl;

        // Each rank sends different amount
        std::vector<int> my_data(rank + 1);
        for (size_t i = 0; i < my_data.size(); ++i) {
            my_data[i] = rank * 100 + i;
        }

        print_data(my_data, "My data", rank);

        // Define receive counts
        std::vector<int> recvcounts(size);
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = i + 1;
        }

        // NEW WAY: Returns vector directly, handles everything!
        auto gathered = comm.allgatherv_vec(my_data, recvcounts);

        comm.barrier();
        if (rank == 0) {
            std::cout << "\nGathered data: ";
            for (auto val : gathered) {
                std::cout << val << " ";
            }
            std::cout << "\n" << std::endl;
        }
    }

    comm.barrier();

    // ========================================
    // Example 4: alltoallv_auto - Personalized exchange
    // ========================================
    {
        std::cout << "\n[Example 4] alltoallv_auto - Personalized all-to-all\n" << std::endl;

        // Each rank sends different amounts to each other rank
        std::vector<int> sendcounts(size);
        std::vector<int> recvcounts(size);

        // Example: rank i sends (i+1) items to rank j
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = rank + 1;
            recvcounts[i] = i + 1;
        }

        int total_send = stComm::Utils::total_size(sendcounts);
        int total_recv = stComm::Utils::total_size(recvcounts);

        std::vector<int> sendbuf(total_send);
        std::vector<int> recvbuf(total_recv);

        // Fill send buffer with rank-specific data
        for (int i = 0; i < total_send; ++i) {
            sendbuf[i] = rank * 1000 + i;
        }

        // OLD WAY: Manually calculate send and receive displacements
        // NEW WAY: Automatic displacement calculation!
        auto req = comm.alltoallv_auto(sendbuf.data(), sendcounts.data(),
                                       recvbuf.data(), recvcounts.data());
        req->wait();

        comm.barrier();
        std::cout << "[Rank " << rank << "] Received: ";
        for (int i = 0; i < std::min(10, total_recv); ++i) {
            std::cout << recvbuf[i] << " ";
        }
        if (total_recv > 10) std::cout << "...";
        std::cout << std::endl;
    }

    comm.barrier();

    // ========================================
    // Example 5: alltoallv_vec - Vector-based all-to-all
    // ========================================
    {
        std::cout << "\n[Example 5] alltoallv_vec - Vector-based all-to-all\n" << std::endl;

        std::vector<int> sendcounts(size);
        std::vector<int> recvcounts(size);

        for (int i = 0; i < size; ++i) {
            sendcounts[i] = 2;  // Send 2 items to each rank
            recvcounts[i] = 2;  // Receive 2 items from each rank
        }

        std::vector<int> sendbuf(size * 2);
        for (size_t i = 0; i < sendbuf.size(); ++i) {
            sendbuf[i] = rank * 100 + i;
        }

        // Vector-based API: handles everything!
        auto recvbuf = comm.alltoallv_vec(sendbuf, sendcounts, recvcounts);

        comm.barrier();
        std::cout << "[Rank " << rank << "] Received: ";
        for (auto val : recvbuf) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    comm.barrier();
    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "All examples completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    stComm::MPIComm::finalize();
    return 0;
}
