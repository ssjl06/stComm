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
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 10000;
    std::vector<int> send_data(N);
    std::vector<int> recv_data(N);

    // Only rank 0 and 1 participate in send/recv
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
    }

    // All ranks participate in MPI_Bcast for synchronization
    MPI_Bcast(send_data.data(), N * sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Only rank 1 verifies
    if (rank == 1) {
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(recv_data[i], send_data[i]) << "Mismatch at index " << i;
        }
    }
}

TEST_F(MPICommTest, SendRecvDouble) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 5000;
    std::vector<double> send_data(N);
    std::vector<double> recv_data(N);

    // Only rank 0 and 1 participate in send/recv
    if (rank == 0) {
        fillRandom(send_data);
        auto req = comm->send(send_data.data(), N, 1, 0);
        req->wait();
    } else if (rank == 1) {
        auto req = comm->recv(recv_data.data(), N, 0, 0);
        req->wait();
    }

    // All ranks participate in MPI_Bcast for synchronization
    MPI_Bcast(send_data.data(), N * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Only rank 1 verifies
    if (rank == 1) {
        for (int i = 0; i < N; ++i) {
            EXPECT_DOUBLE_EQ(recv_data[i], send_data[i]) << "Mismatch at index " << i;
        }
    }
}

TEST_F(MPICommTest, SendRecvFloat) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 8000;
    std::vector<float> send_data(N);
    std::vector<float> recv_data(N);

    // Only rank 0 and 1 participate in send/recv
    if (rank == 0) {
        fillRandom(send_data);
        auto req = comm->send(send_data.data(), N, 1, 0);
        req->wait();
    } else if (rank == 1) {
        auto req = comm->recv(recv_data.data(), N, 0, 0);
        req->wait();
    }

    // All ranks participate in MPI_Bcast for synchronization
    MPI_Bcast(send_data.data(), N * sizeof(float), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Only rank 1 verifies
    if (rank == 1) {
        for (int i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(recv_data[i], send_data[i]) << "Mismatch at index " << i;
        }
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
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 1000;
    std::vector<int8_t> send_data(N);
    std::vector<int8_t> recv_data(N);

    // Only rank 0 and 1 participate in send/recv
    if (rank == 0) {
        fillRandom(send_data);
        comm->send(send_data.data(), N, 1, 0)->wait();
    } else if (rank == 1) {
        comm->recv(recv_data.data(), N, 0, 0)->wait();
    }

    // All ranks participate in MPI_Bcast for synchronization
    MPI_Bcast(send_data.data(), N, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Only rank 1 verifies
    if (rank == 1) {
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(recv_data[i], send_data[i]);
        }
    }
}

TEST_F(MPICommTest, SendRecvInt64) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    const int N = 2000;
    std::vector<int64_t> send_data(N);
    std::vector<int64_t> recv_data(N);

    // Only rank 0 and 1 participate in send/recv
    if (rank == 0) {
        fillRandom(send_data);
        comm->send(send_data.data(), N, 1, 0)->wait();
    } else if (rank == 1) {
        comm->recv(recv_data.data(), N, 0, 0)->wait();
    }

    // All ranks participate in MPI_Bcast for synchronization
    MPI_Bcast(send_data.data(), N * sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Only rank 1 verifies
    if (rank == 1) {
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(recv_data[i], send_data[i]);
        }
    }
}

// ============================================================================
// Bcast tests
// ============================================================================

TEST_F(MPICommTest, BcastInt) {
    const int N = 100;
    std::vector<int> buf(N);
    if (rank == 0) {
        for (int i = 0; i < N; ++i) buf[i] = i * 7 + 1;
    }
    auto req = comm->bcast(buf.data(), N, 0);
    req->wait();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(buf[i], i * 7 + 1) << "rank " << rank << " mismatch at " << i;
    }
}

TEST_F(MPICommTest, BcastDouble) {
    const int N = 64;
    std::vector<double> buf(N);
    if (rank == 0) {
        for (int i = 0; i < N; ++i) buf[i] = 1.5 * i;
    }
    auto req = comm->bcast(buf.data(), N, 0);
    req->wait();
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(buf[i], 1.5 * i);
    }
}

TEST_F(MPICommTest, BcastFromNonRootZero) {
    // Broadcast from a non-zero root (only meaningful when size >= 2).
    if (size < 2) GTEST_SKIP();
    const int N = 50;
    const int root = size - 1;
    std::vector<int64_t> buf(N, -1);
    if (rank == root) {
        for (int i = 0; i < N; ++i) buf[i] = i * 13;
    }
    auto req = comm->bcast(buf.data(), N, root);
    req->wait();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(buf[i], i * 13);
    }
}

// ============================================================================
// AllreduceMaxloc tests
// ============================================================================

TEST_F(MPICommTest, AllreduceMaxlocLong) {
    // Each rank reports a distinct value. Highest belongs to rank (size-1).
    const long my_value = 100L + rank;
    auto result = comm->allreduceMaxloc<long>(my_value);
    EXPECT_EQ(result.first,  100L + (size - 1));
    EXPECT_EQ(result.second, size - 1);
}

TEST_F(MPICommTest, AllreduceMaxlocInt64) {
    // int64_t goes through `long` on LP64.
    const int64_t my_value = 1000000000LL + rank * 1000000LL;
    auto result = comm->allreduceMaxloc<int64_t>(my_value);
    EXPECT_EQ(result.first,  1000000000LL + (size - 1) * 1000000LL);
    EXPECT_EQ(result.second, size - 1);
}

TEST_F(MPICommTest, AllreduceMaxlocTiesPickSmallerRank) {
    // All ranks report the same value → MPI_MAXLOC tie-break = smaller rank wins.
    auto result = comm->allreduceMaxloc<long>(42L);
    EXPECT_EQ(result.first,  42L);
    EXPECT_EQ(result.second, 0);
}

TEST_F(MPICommTest, AllreduceMaxlocFloat) {
    const float my_value = 0.5f + static_cast<float>(rank);
    auto result = comm->allreduceMaxloc<float>(my_value);
    EXPECT_FLOAT_EQ(result.first,  0.5f + static_cast<float>(size - 1));
    EXPECT_EQ(result.second, size - 1);
}

TEST_F(MPICommTest, AllreduceMaxlocDouble) {
    const double my_value = 1.25 * (rank + 1);
    auto result = comm->allreduceMaxloc<double>(my_value);
    EXPECT_DOUBLE_EQ(result.first,  1.25 * size);
    EXPECT_EQ(result.second, size - 1);
}

// ============================================================================
// Exscan tests
// ============================================================================

TEST_F(MPICommTest, ExscanSumInt64) {
    // Each rank contributes (rank+1). Exclusive prefix sum at rank r is
    // sum_{k=0..r-1}(k+1) = r*(r+1)/2. Rank 0 gets identity (0).
    const int64_t my_value = rank + 1;
    int64_t result = comm->exscan<int64_t>(my_value, MPI_SUM);
    const int64_t expected = static_cast<int64_t>(rank) * (rank + 1) / 2;
    EXPECT_EQ(result, expected);
}

TEST_F(MPICommTest, ExscanSumUint64) {
    const uint64_t my_value = 100ULL + rank;
    uint64_t result = comm->exscan<uint64_t>(my_value, MPI_SUM);
    // exclusive prefix: sum_{k=0..r-1}(100 + k) = r*100 + r*(r-1)/2
    const uint64_t expected = static_cast<uint64_t>(rank) * 100ULL
                            + static_cast<uint64_t>(rank) * (rank - 1) / 2;
    EXPECT_EQ(result, expected);
}

TEST_F(MPICommTest, ExscanRank0GetsIdentity) {
    // For rank 0, exscan should return the operator identity (0 for SUM).
    int64_t result = comm->exscan<int64_t>(999, MPI_SUM);
    if (rank == 0) {
        EXPECT_EQ(result, 0);
    }
}

TEST_F(MPICommTest, ExscanMaxInt32) {
    // Each rank contributes its own rank. MAX exclusive prefix at rank r is r-1 (or
    // the MIN possible for rank 0).
    int32_t my_value = rank;
    int32_t result = comm->exscan<int32_t>(my_value, MPI_MAX);
    if (rank == 0) {
        // Identity behavior: we return T{} which is 0. MPI doesn't define the
        // identity for MAX in general, but our implementation returns T{}.
        EXPECT_EQ(result, 0);
    } else {
        EXPECT_EQ(result, rank - 1);
    }
}
