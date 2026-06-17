/**
 * @file mpi_large_data.cpp
 * @brief Large (>2GB) transfer via the stComm::Comm facade.
 *
 * Demonstrates:
 * - Automatic chunking for data larger than 2GB (transparent, no API change)
 * - send/recv reached through the comm.mpi() escape hatch; the MPIRequest it
 *   returns transparently tracks the multiple chunked MPI_Request handles
 */

#include "stComm/stComm.h"
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {
    stComm::Comm::initialize(&argc, &argv);

    stComm::Comm comm;                 // host-only Comm
    int rank = comm.getRank();
    int size = comm.getSize();

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "This example requires at least 2 processes" << std::endl;
        }
        stComm::Comm::finalize();
        return 1;
    }

    // Test with large data: 3GB = 3 * 1024 * 1024 * 1024 / 8 doubles
    const size_t GB = 1024ULL * 1024ULL * 1024ULL;
    const size_t data_size_bytes = 3 * GB;  // 3GB
    const size_t count = data_size_bytes / sizeof(double);

    std::cout << "Rank " << rank << ": Preparing " << (data_size_bytes / GB)
              << " GB of data (" << count << " doubles)" << std::endl;

    if (rank == 0) {
        // Rank 0 sends large data to rank 1
        std::vector<double> data(count);

        // Initialize with pattern for verification
        std::cout << "Rank 0: Initializing data..." << std::endl;
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<double>(i % 1000000);
        }

        std::cout << "Rank 0: Sending " << (data_size_bytes / GB)
                  << " GB to rank 1 (automatic chunking)..." << std::endl;
        // p2p send via the MPI escape hatch; chunking happens inside MPIComm.
        comm.mpi().send(data.data(), count, 1, 0)->wait();
        std::cout << "Rank 0: Send complete!" << std::endl;

    } else if (rank == 1) {
        // Rank 1 receives large data from rank 0
        std::vector<double> data(count);

        std::cout << "Rank 1: Receiving " << (data_size_bytes / GB)
                  << " GB from rank 0..." << std::endl;
        comm.mpi().recv(data.data(), count, 0, 0)->wait();
        std::cout << "Rank 1: Receive complete!" << std::endl;

        // Verify data (check first, middle, and last elements)
        std::cout << "Rank 1: Verifying data..." << std::endl;
        bool verified = true;

        // Check first 1000 elements
        for (size_t i = 0; i < 1000 && i < count; ++i) {
            double expected = static_cast<double>(i % 1000000);
            if (data[i] != expected) {
                std::cerr << "Verification failed at index " << i
                          << ": expected " << expected << ", got " << data[i] << std::endl;
                verified = false;
                break;
            }
        }

        // Check middle 1000 elements
        if (verified && count > 2000) {
            size_t mid = count / 2;
            for (size_t i = mid; i < mid + 1000 && i < count; ++i) {
                double expected = static_cast<double>(i % 1000000);
                if (data[i] != expected) {
                    std::cerr << "Verification failed at index " << i
                              << ": expected " << expected << ", got " << data[i] << std::endl;
                    verified = false;
                    break;
                }
            }
        }

        // Check last 1000 elements
        if (verified && count > 1000) {
            for (size_t i = count - 1000; i < count; ++i) {
                double expected = static_cast<double>(i % 1000000);
                if (data[i] != expected) {
                    std::cerr << "Verification failed at index " << i
                              << ": expected " << expected << ", got " << data[i] << std::endl;
                    verified = false;
                    break;
                }
            }
        }

        if (verified) {
            std::cout << "Rank 1: Data verification successful!" << std::endl;
        } else {
            std::cerr << "Rank 1: Data verification FAILED!" << std::endl;
        }
    }

    std::cout << "Rank " << rank << ": Example completed" << std::endl;

    stComm::Comm::finalize();
    return 0;
}
