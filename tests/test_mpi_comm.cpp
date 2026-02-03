/**
 * @file test_mpi_comm.cpp
 * @brief Comprehensive tests for MPIComm class with random data
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

class MPICommTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm = std::make_unique<stComm::MPIComm>();
        rank = comm->getRank();
        size = comm->getSize();

        // Initialize random generator with rank-based seed for reproducibility
        rng.seed(12345 + rank);
    }

    void TearDown() override {
        comm.reset();
    }

    // Helper to generate random data
    template<typename T>
    void fillRandom(std::vector<T>& data) {
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(),
                                                   std::numeric_limits<T>::max());
            for (auto& val : data) {
                val = dist(rng);
            }
        } else {
            std::uniform_real_distribution<T> dist(-1000.0, 1000.0);
            for (auto& val : data) {
                val = dist(rng);
            }
        }
    }

    std::unique_ptr<stComm::MPIComm> comm;
    int rank;
    int size;
    std::mt19937 rng;
};

// ============================================================================
// Basic Interface Tests
// ============================================================================

TEST_F(MPICommTest, GetRank) {
    int r = comm->getRank();
    EXPECT_GE(r, 0);
    EXPECT_LT(r, size);
    EXPECT_EQ(r, rank);
}

TEST_F(MPICommTest, GetSize) {
    int s = comm->getSize();
    EXPECT_GT(s, 0);
    EXPECT_EQ(s, size);
}

TEST_F(MPICommTest, GetBackend) {
    EXPECT_EQ(comm->getBackend(), stComm::Backend::MPI);
}

TEST_F(MPICommTest, GetHandle) {
    MPI_Comm handle = comm->getHandle();
    EXPECT_NE(handle, MPI_COMM_NULL);
}

TEST_F(MPICommTest, Barrier) {
    // Test multiple barriers
    EXPECT_NO_THROW(comm->barrier());
    EXPECT_NO_THROW(comm->barrier());
    EXPECT_NO_THROW(comm->barrier());
}

// ============================================================================
// Point-to-Point Communication Tests
// ============================================================================

TEST_F(MPICommTest, SendRecvInt) {
    if (size != 2) {
        GTEST_SKIP() << "Test requires exactly 2 processes (uses MPI_Bcast for verification)";
    }

    const int N = 10000;
    std::vector<int> send_data(N);
    std::vector<int> recv_data(N);

    if (rank == 0) {
        fillRandom(send_data);
        auto req = comm->send(send_data.data(), N, 1, 0);
        EXPECT_NE(req, nullptr);
        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);
    } else if (rank == 1) {
        auto req = comm->recv(recv_data.data(), N, 0, 0);
        EXPECT_NE(req, nullptr);
        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

        // Broadcast original data for verification
        MPI_Bcast(send_data.data(), N * sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Verify
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(recv_data[i], send_data[i]) << "Mismatch at index " << i;
        }
    }

    if (rank == 0) {
        MPI_Bcast(send_data.data(), N * sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
}

TEST_F(MPICommTest, SendRecvDouble) {
    if (size != 2) {
        GTEST_SKIP() << "Test requires exactly 2 processes (uses MPI_Bcast for verification)";
    }

    const int N = 5000;
    std::vector<double> send_data(N);
    std::vector<double> recv_data(N);

    if (rank == 0) {
        fillRandom(send_data);
        auto req = comm->send(send_data.data(), N, 1, 0);
        req->wait();
    } else if (rank == 1) {
        auto req = comm->recv(recv_data.data(), N, 0, 0);
        req->wait();

        MPI_Bcast(send_data.data(), N * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < N; ++i) {
            EXPECT_DOUBLE_EQ(recv_data[i], send_data[i]) << "Mismatch at index " << i;
        }
    }

    if (rank == 0) {
        MPI_Bcast(send_data.data(), N * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
}

TEST_F(MPICommTest, SendRecvFloat) {
    if (size != 2) {
        GTEST_SKIP() << "Test requires exactly 2 processes (uses MPI_Bcast for verification)";
    }

    const int N = 8000;
    std::vector<float> send_data(N);
    std::vector<float> recv_data(N);

    if (rank == 0) {
        fillRandom(send_data);
        auto req = comm->send(send_data.data(), N, 1, 0);
        req->wait();
    } else if (rank == 1) {
        auto req = comm->recv(recv_data.data(), N, 0, 0);
        req->wait();

        MPI_Bcast(send_data.data(), N * sizeof(float), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(recv_data[i], send_data[i]) << "Mismatch at index " << i;
        }
    }

    if (rank == 0) {
        MPI_Bcast(send_data.data(), N * sizeof(float), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
}

TEST_F(MPICommTest, SendRecvRingPattern) {
    const int N = 1000;
    std::vector<int> send_data(N);
    std::vector<int> recv_data(N);

    fillRandom(send_data);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    auto send_req = comm->send(send_data.data(), N, next, 0);
    auto recv_req = comm->recv(recv_data.data(), N, prev, 0);

    send_req->wait();
    recv_req->wait();

    // Gather all data to verify ring pattern
    std::vector<int> all_data(N * size);
    MPI_Allgather(send_data.data(), N * sizeof(int), MPI_BYTE,
                  all_data.data(), N * sizeof(int), MPI_BYTE, MPI_COMM_WORLD);

    // Verify received from previous rank
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(recv_data[i], all_data[prev * N + i]) << "Ring pattern failed at index " << i;
    }
}

// ============================================================================
// Large Data Transfer Tests (>2GB)
// ============================================================================

TEST_F(MPICommTest, SendRecvLargeData) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // 3GB worth of integers
    const size_t GB = 1024ULL * 1024ULL * 1024ULL;
    const size_t total_bytes = 3 * GB;
    const size_t count = total_bytes / sizeof(int);

    // Helper lambda to generate deterministic pattern (avoids MPI_Bcast >2GB limitation)
    auto generatePattern = [](size_t i) -> int {
        return static_cast<int>((i * 7919ULL + 104729ULL) % 1000000);
    };

    if (rank == 0) {
        std::vector<int> send_data(count);

        // Fill with deterministic pattern
        for (size_t i = 0; i < count; ++i) {
            send_data[i] = generatePattern(i);
        }

        auto req = comm->send(send_data.data(), count, 1, 0);
        EXPECT_NE(req, nullptr);
        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

    } else if (rank == 1) {
        std::vector<int> recv_data(count);

        auto req = comm->recv(recv_data.data(), count, 0, 0);
        EXPECT_NE(req, nullptr);
        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

        // Verify sampling (check every 1M elements to save time)
        const size_t stride = 1000000;
        for (size_t i = 0; i < count; i += stride) {
            EXPECT_EQ(recv_data[i], generatePattern(i))
                << "Large data mismatch at index " << i;
        }

        // Verify first 1000 elements
        for (size_t i = 0; i < 1000 && i < count; ++i) {
            EXPECT_EQ(recv_data[i], generatePattern(i)) << "Mismatch at index " << i;
        }

        // Verify middle 1000 elements
        size_t mid = count / 2;
        for (size_t i = mid; i < mid + 1000 && i < count; ++i) {
            EXPECT_EQ(recv_data[i], generatePattern(i)) << "Mismatch at index " << i;
        }

        // Verify last 1000 elements
        for (size_t i = count - 1000; i < count; ++i) {
            EXPECT_EQ(recv_data[i], generatePattern(i)) << "Mismatch at index " << i;
        }
    }
}

// ============================================================================
// Collective Communication Tests
// ============================================================================

TEST_F(MPICommTest, AllgathervUniformCounts) {
    const int sendcount = 100;
    std::vector<int> sendbuf(sendcount);
    fillRandom(sendbuf);

    std::vector<int> recvcounts(size, sendcount);
    int total_recv = stComm::Utils::totalSize(recvcounts);
    std::vector<int> recvbuf(total_recv);

    auto req = comm->allgatherv(sendbuf.data(), sendcount, recvbuf.data(), recvcounts.data());
    EXPECT_NE(req, nullptr);
    req->wait();
    EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

    // Verify by comparing with MPI_Allgather
    std::vector<int> expected(total_recv);
    MPI_Allgather(sendbuf.data(), sendcount * sizeof(int), MPI_BYTE,
                  expected.data(), sendcount * sizeof(int), MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(recvbuf[i], expected[i]) << "Allgatherv mismatch at index " << i;
    }
}

TEST_F(MPICommTest, AllgathervVariableCounts) {
    // Each rank sends different amount
    int sendcount = (rank + 1) * 50;
    std::vector<int> sendbuf(sendcount);
    fillRandom(sendbuf);

    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (i + 1) * 50;
    }

    int total_recv = stComm::Utils::totalSize(recvcounts);
    std::vector<int> recvbuf(total_recv);

    auto req = comm->allgatherv(sendbuf.data(), sendcount, recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify with manual allgather
    auto displs = stComm::Utils::calculateDisplacements(recvcounts);
    std::vector<int> expected(total_recv);

    std::vector<int> byte_recvcounts(size);
    std::vector<int> byte_displs(size);
    for (int i = 0; i < size; ++i) {
        byte_recvcounts[i] = recvcounts[i] * sizeof(int);
        byte_displs[i] = displs[i] * sizeof(int);
    }

    MPI_Allgatherv(sendbuf.data(), sendcount * sizeof(int), MPI_BYTE,
                   expected.data(), byte_recvcounts.data(), byte_displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(recvbuf[i], expected[i]) << "Variable allgatherv mismatch at index " << i;
    }
}

TEST_F(MPICommTest, AlltoallvUniformCounts) {
    const int count_per_rank = 50;
    std::vector<int> sendcounts(size, count_per_rank);
    std::vector<int> recvcounts(size, count_per_rank);

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    std::vector<int> sendbuf(total_send);
    std::vector<int> recvbuf(total_recv);

    fillRandom(sendbuf);

    auto req = comm->alltoallv(sendbuf.data(), sendcounts.data(),
                               recvbuf.data(), recvcounts.data());
    EXPECT_NE(req, nullptr);
    req->wait();
    EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

    // Verify with MPI_Alltoall
    std::vector<int> expected(total_recv);
    MPI_Alltoall(sendbuf.data(), count_per_rank * sizeof(int), MPI_BYTE,
                 expected.data(), count_per_rank * sizeof(int), MPI_BYTE,
                 MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(recvbuf[i], expected[i]) << "Alltoallv uniform mismatch at index " << i;
    }
}

TEST_F(MPICommTest, AlltoallvVariableCounts) {
    std::vector<int> sendcounts(size);
    std::vector<int> recvcounts(size);

    // Rank i sends (i+1)*10 to rank j, receives (j+1)*10 from rank j
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rank + 1) * 10;
        recvcounts[i] = (i + 1) * 10;
    }

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    std::vector<int> sendbuf(total_send);
    std::vector<int> recvbuf(total_recv);

    fillRandom(sendbuf);

    auto req = comm->alltoallv(sendbuf.data(), sendcounts.data(),
                               recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify with MPI_Alltoallv
    auto sdispls = stComm::Utils::calculateDisplacements(sendcounts);
    auto rdispls = stComm::Utils::calculateDisplacements(recvcounts);

    std::vector<int> byte_sendcounts(size);
    std::vector<int> byte_sdispls(size);
    std::vector<int> byte_recvcounts(size);
    std::vector<int> byte_rdispls(size);

    for (int i = 0; i < size; ++i) {
        byte_sendcounts[i] = sendcounts[i] * sizeof(int);
        byte_sdispls[i] = sdispls[i] * sizeof(int);
        byte_recvcounts[i] = recvcounts[i] * sizeof(int);
        byte_rdispls[i] = rdispls[i] * sizeof(int);
    }

    std::vector<int> expected(total_recv);
    MPI_Alltoallv(sendbuf.data(), byte_sendcounts.data(), byte_sdispls.data(), MPI_BYTE,
                  expected.data(), byte_recvcounts.data(), byte_rdispls.data(), MPI_BYTE,
                  MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(recvbuf[i], expected[i]) << "Variable alltoallv mismatch at index " << i;
    }
}

// ============================================================================
// Async Operation Tests
// ============================================================================

TEST_F(MPICommTest, AsyncSendRecv) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 5000;
    std::vector<int> data(N);

    if (rank == 0) {
        fillRandom(data);
        auto req = comm->send(data.data(), N, 1, 0);

        // Do some work while send is in progress
        volatile int dummy = 0;
        for (int i = 0; i < 10000; ++i) {
            dummy += i;
        }

        // Test status before wait
        req->test();

        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

    } else if (rank == 1) {
        auto req = comm->recv(data.data(), N, 0, 0);

        // Test multiple times
        req->test();
        req->test();

        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);
    }
}

// ============================================================================
// Multiple Data Type Tests
// ============================================================================

TEST_F(MPICommTest, SendRecvInt8) {
    if (size != 2) {
        GTEST_SKIP() << "Test requires exactly 2 processes (uses MPI_Bcast for verification)";
    }

    const int N = 1000;
    std::vector<int8_t> send_data(N);
    std::vector<int8_t> recv_data(N);

    if (rank == 0) {
        fillRandom(send_data);
        comm->send(send_data.data(), N, 1, 0)->wait();
    } else if (rank == 1) {
        comm->recv(recv_data.data(), N, 0, 0)->wait();
        MPI_Bcast(send_data.data(), N, MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(recv_data[i], send_data[i]);
        }
    }

    if (rank == 0) {
        MPI_Bcast(send_data.data(), N, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
}

TEST_F(MPICommTest, SendRecvInt64) {
    if (size != 2) {
        GTEST_SKIP() << "Test requires exactly 2 processes (uses MPI_Bcast for verification)";
    }

    const int N = 2000;
    std::vector<int64_t> send_data(N);
    std::vector<int64_t> recv_data(N);

    if (rank == 0) {
        fillRandom(send_data);
        comm->send(send_data.data(), N, 1, 0)->wait();
    } else if (rank == 1) {
        comm->recv(recv_data.data(), N, 0, 0)->wait();
        MPI_Bcast(send_data.data(), N * sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(recv_data[i], send_data[i]);
        }
    }

    if (rank == 0) {
        MPI_Bcast(send_data.data(), N * sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
}
