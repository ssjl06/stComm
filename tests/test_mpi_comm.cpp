/**
 * @file test_mpi_comm.cpp
 * @brief Comprehensive tests for MPIComm class with random data
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include "test_util.h"
#include <vector>
#include <random>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

using stComm_test::uz;

class MPICommTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm = std::make_unique<stComm::MPIComm>();
        rank = comm->getRank();
        size = comm->getSize();

        // Initialize random generator with rank-based seed for reproducibility
        rng.seed(static_cast<std::mt19937::result_type>(12345 + rank));
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
            EXPECT_EQ(recv_data[uz(i)], send_data[uz(i)]) << "Mismatch at index " << i;
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
            EXPECT_DOUBLE_EQ(recv_data[uz(i)], send_data[uz(i)]) << "Mismatch at index " << i;
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
            EXPECT_FLOAT_EQ(recv_data[uz(i)], send_data[uz(i)]) << "Mismatch at index " << i;
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
    std::vector<int> all_data(uz(N * size));
    MPI_Allgather(send_data.data(), N * sizeof(int), MPI_BYTE,
                  all_data.data(), N * sizeof(int), MPI_BYTE, MPI_COMM_WORLD);

    // Verify received from previous rank
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(recv_data[uz(i)], all_data[uz(prev * N + i)]) << "Ring pattern failed at index " << i;
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

    std::vector<int> recvcounts(uz(size), sendcount);
    int total_recv = stComm::Utils::totalSize(recvcounts);
    std::vector<int> recvbuf(uz(total_recv));

    auto req = comm->allgatherv(sendbuf.data(), sendcount, recvbuf.data(), recvcounts.data());
    EXPECT_NE(req, nullptr);
    req->wait();
    EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

    // Verify by comparing with MPI_Allgather
    std::vector<int> expected(uz(total_recv));
    MPI_Allgather(sendbuf.data(), sendcount * sizeof(int), MPI_BYTE,
                  expected.data(), sendcount * sizeof(int), MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(recvbuf[uz(i)], expected[uz(i)]) << "Allgatherv mismatch at index " << i;
    }
}

TEST_F(MPICommTest, AllgathervVariableCounts) {
    // Each rank sends different amount
    int sendcount = (rank + 1) * 50;
    std::vector<int> sendbuf(uz(sendcount));
    fillRandom(sendbuf);

    std::vector<int> recvcounts(uz(size));
    for (int i = 0; i < size; ++i) {
        recvcounts[uz(i)] = (i + 1) * 50;
    }

    int total_recv = stComm::Utils::totalSize(recvcounts);
    std::vector<int> recvbuf(uz(total_recv));

    auto req = comm->allgatherv(sendbuf.data(), sendcount, recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify with manual allgather
    auto displs = stComm::Utils::calculateDisplacements(recvcounts);
    std::vector<int> expected(uz(total_recv));

    std::vector<int> byte_recvcounts(uz(size));
    std::vector<int> byte_displs(uz(size));
    for (int i = 0; i < size; ++i) {
        byte_recvcounts[uz(i)] = recvcounts[uz(i)] * static_cast<int>(sizeof(int));
        byte_displs[uz(i)] = displs[uz(i)] * static_cast<int>(sizeof(int));
    }

    MPI_Allgatherv(sendbuf.data(), sendcount * static_cast<int>(sizeof(int)), MPI_BYTE,
                   expected.data(), byte_recvcounts.data(), byte_displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(recvbuf[uz(i)], expected[uz(i)]) << "Variable allgatherv mismatch at index " << i;
    }
}

TEST_F(MPICommTest, AlltoallvUniformCounts) {
    const int count_per_rank = 50;
    std::vector<int> sendcounts(uz(size), count_per_rank);
    std::vector<int> recvcounts(uz(size), count_per_rank);

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    std::vector<int> sendbuf(uz(total_send));
    std::vector<int> recvbuf(uz(total_recv));

    fillRandom(sendbuf);

    auto req = comm->alltoallv(sendbuf.data(), sendcounts.data(),
                               recvbuf.data(), recvcounts.data());
    EXPECT_NE(req, nullptr);
    req->wait();
    EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

    // Verify with MPI_Alltoall
    std::vector<int> expected(uz(total_recv));
    MPI_Alltoall(sendbuf.data(), count_per_rank * sizeof(int), MPI_BYTE,
                 expected.data(), count_per_rank * sizeof(int), MPI_BYTE,
                 MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(recvbuf[uz(i)], expected[uz(i)]) << "Alltoallv uniform mismatch at index " << i;
    }
}

TEST_F(MPICommTest, AlltoallvVariableCounts) {
    std::vector<int> sendcounts(uz(size));
    std::vector<int> recvcounts(uz(size));

    // Rank i sends (i+1)*10 to rank j, receives (j+1)*10 from rank j
    for (int i = 0; i < size; ++i) {
        sendcounts[uz(i)] = (rank + 1) * 10;
        recvcounts[uz(i)] = (i + 1) * 10;
    }

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    std::vector<int> sendbuf(uz(total_send));
    std::vector<int> recvbuf(uz(total_recv));

    fillRandom(sendbuf);

    auto req = comm->alltoallv(sendbuf.data(), sendcounts.data(),
                               recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify with MPI_Alltoallv
    auto sdispls = stComm::Utils::calculateDisplacements(sendcounts);
    auto rdispls = stComm::Utils::calculateDisplacements(recvcounts);

    std::vector<int> byte_sendcounts(uz(size));
    std::vector<int> byte_sdispls(uz(size));
    std::vector<int> byte_recvcounts(uz(size));
    std::vector<int> byte_rdispls(uz(size));

    for (int i = 0; i < size; ++i) {
        byte_sendcounts[uz(i)] = sendcounts[uz(i)] * static_cast<int>(sizeof(int));
        byte_sdispls[uz(i)] = sdispls[uz(i)] * static_cast<int>(sizeof(int));
        byte_recvcounts[uz(i)] = recvcounts[uz(i)] * static_cast<int>(sizeof(int));
        byte_rdispls[uz(i)] = rdispls[uz(i)] * static_cast<int>(sizeof(int));
    }

    std::vector<int> expected(uz(total_recv));
    MPI_Alltoallv(sendbuf.data(), byte_sendcounts.data(), byte_sdispls.data(), MPI_BYTE,
                  expected.data(), byte_recvcounts.data(), byte_rdispls.data(), MPI_BYTE,
                  MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; ++i) {
        EXPECT_EQ(recvbuf[uz(i)], expected[uz(i)]) << "Variable alltoallv mismatch at index " << i;
    }
}

// ============================================================================
// Async Operation Tests
// ============================================================================

// The v-collectives take int element counts and int element displacements. A
// layout outside that range must throw before any MPI call rather than wrap
// into a wrong-size transfer. Every rank passes the same bad counts, so all
// throw together (no buffers are touched — nullptr is fine).
TEST_F(MPICommTest, VCollectivesRejectBadLayout) {
    std::vector<int> negative(uz(size), -1);
    EXPECT_THROW(comm->alltoallv<std::int64_t>(nullptr, negative.data(),
                                               nullptr, negative.data()),
                 std::overflow_error);
    EXPECT_THROW(comm->allgatherv<std::int64_t>(nullptr, 0,
                                                nullptr, negative.data()),
                 std::overflow_error);

    if (size < 3) return;  // an element offset past INT_MAX needs a third rank
    std::vector<int> huge(uz(size), INT_MAX);
    EXPECT_THROW(comm->alltoallv<std::int64_t>(nullptr, huge.data(),
                                               nullptr, huge.data()),
                 std::overflow_error);
    EXPECT_THROW(comm->allgatherv<std::int64_t>(nullptr, 0,
                                                nullptr, huge.data()),
                 std::overflow_error);
}

// Elements wider than a primitive travel as one contiguous-bytes datatype with
// element counts (not as sizeof(T)-scaled byte counts).
TEST_F(MPICommTest, AlltoallvStructElements) {
    struct Rec { std::int64_t id; double w; std::int32_t tag; };
    const int per = 3;
    std::vector<int> counts(uz(size), per);
    std::vector<Rec> send(uz(size * per)), recv(uz(size * per));
    for (int j = 0; j < size; ++j)
        for (int k = 0; k < per; ++k)
            send[uz(j * per + k)] = Rec{rank * 100 + j, 0.5 * k, k};

    comm->alltoallv(send.data(), counts.data(), recv.data(), counts.data())->wait();

    for (int i = 0; i < size; ++i)
        for (int k = 0; k < per; ++k) {
            const Rec& r = recv[uz(i * per + k)];
            EXPECT_EQ(r.id, i * 100 + rank);
            EXPECT_DOUBLE_EQ(r.w, 0.5 * k);
            EXPECT_EQ(r.tag, k);
        }
}

// Unsigned 64-bit MAXLOC must compare as unsigned: a value >= 2^63 used to be
// cast to a negative long and lose to any small value.
TEST_F(MPICommTest, AllreduceMaxlocUnsigned64) {
    const std::uint64_t big = 0x8000000000000001ULL;
    const std::uint64_t mine = (rank == 0) ? big : 5ULL;

    const auto blocking = comm->allreduceMaxloc(mine);
    EXPECT_EQ(blocking.first, big);
    EXPECT_EQ(blocking.second, 0);

    std::pair<std::uint64_t, int> async{0, -1};
    comm->allreduceMaxloc(mine, &async)->wait();
    EXPECT_EQ(async.first, big);
    EXPECT_EQ(async.second, 0);

    // Signed 64-bit keeps its own order (negative stays below positive).
    const std::int64_t smine = (rank == 0) ? std::int64_t{7} : std::int64_t{-9};
    const auto s = comm->allreduceMaxloc(smine);
    EXPECT_EQ(s.first, 7);
    EXPECT_EQ(s.second, 0);
}

TEST_F(MPICommTest, AllreduceInPlace) {
    const int n = 4;
    std::vector<std::int64_t> buf(uz(n));
    for (int i = 0; i < n; ++i) buf[uz(i)] = rank + i;

    comm->allreduce(buf.data(), buf.data(), uz(n), MPI_SUM)->wait();

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(buf[uz(i)], static_cast<std::int64_t>(size) * (size - 1) / 2 +
                              static_cast<std::int64_t>(size) * i);
    }
}

// A request dropped without wait() must still let the collective finish (the
// destructor waits) instead of freeing the arrays MPI is reading.
TEST_F(MPICommTest, DroppedRequestStillCompletes) {
    const int per = 2;
    std::vector<int> counts(uz(size), per);
    std::vector<int> send(uz(per), rank), recv(uz(size * per), -1);

    comm->allgatherv(send.data(), per, recv.data(), counts.data());  // dropped

    for (int i = 0; i < size; ++i)
        for (int k = 0; k < per; ++k) EXPECT_EQ(recv[uz(i * per + k)], i);

    std::int64_t out = -1;
    comm->exscan(std::int64_t{1}, &out);  // dropped: finalizer must not run
    EXPECT_EQ(out, -1);
}

// ============================================================================
// size_t-count collectives (no INT_MAX limit; chunked point-to-point)
// ============================================================================

namespace {

// Shrink the point-to-point chunk so tiny payloads span many messages — the
// same code path a >1 GiB transfer takes. Every rank runs the same test, so
// all use the same value.
struct ChunkBytesGuard {
    std::size_t saved;
    explicit ChunkBytesGuard(std::size_t bytes) : saved(stComm::detail::p2pChunkBytes()) {
        stComm::detail::p2pChunkBytes() = bytes;
    }
    ~ChunkBytesGuard() { stComm::detail::p2pChunkBytes() = saved; }
};

// Rank s sends rank r this many elements (0 for some pairs, on purpose).
std::size_t pairCount(int s, int r) { return static_cast<std::size_t>((s * 3 + r * 5) % 7); }

std::int64_t pairValue(int s, int r, std::size_t k) {
    return static_cast<std::int64_t>(s) * 1000000 + r * 1000 + static_cast<std::int64_t>(k);
}

}  // namespace

TEST_F(MPICommTest, AlltoallvSizeCountsChunked) {
    // 12 B chunks under 8 B elements: chunk boundaries fall inside elements.
    for (std::size_t chunk : {std::size_t{12}, std::size_t{1} << 30}) {
        const ChunkBytesGuard guard(chunk);
        std::vector<std::size_t> sendcounts(uz(size)), recvcounts(uz(size));
        std::size_t total_send = 0, total_recv = 0;
        for (int r = 0; r < size; ++r) {
            sendcounts[uz(r)] = pairCount(rank, r);
            recvcounts[uz(r)] = pairCount(r, rank);
            total_send += sendcounts[uz(r)];
            total_recv += recvcounts[uz(r)];
        }
        std::vector<std::int64_t> send(total_send), recv(total_recv, -1);
        std::size_t at = 0;
        for (int r = 0; r < size; ++r)
            for (std::size_t k = 0; k < sendcounts[uz(r)]; ++k) send[at++] = pairValue(rank, r, k);

        auto req = comm->alltoallv(send.data(), sendcounts.data(),
                                   recv.data(), recvcounts.data());
        if (chunk == 12) {
            // One message per 12 B (rounded up) per non-empty pair, both ways.
            std::size_t expected = 0;
            for (int r = 0; r < size; ++r) {
                if (r == rank) continue;
                expected += (sendcounts[uz(r)] * 8 + 11) / 12;
                expected += (recvcounts[uz(r)] * 8 + 11) / 12;
            }
            EXPECT_EQ(req->getHandles().size(), expected);
        }
        req->wait();

        at = 0;
        for (int s = 0; s < size; ++s)
            for (std::size_t k = 0; k < recvcounts[uz(s)]; ++k)
                EXPECT_EQ(recv[at++], pairValue(s, rank, k)) << "from " << s << " elem " << k;
    }
}

TEST_F(MPICommTest, AllgathervSizeCountsChunked) {
    const ChunkBytesGuard guard(20);
    std::vector<std::size_t> recvcounts(uz(size));
    std::size_t total = 0;
    for (int r = 0; r < size; ++r) {
        recvcounts[uz(r)] = static_cast<std::size_t>((r * 4) % 9);  // rank 0 sends nothing
        total += recvcounts[uz(r)];
    }
    const std::size_t mine = recvcounts[uz(rank)];
    std::vector<std::int64_t> send(mine), recv(total, -1);
    for (std::size_t k = 0; k < mine; ++k) send[k] = pairValue(rank, 0, k);

    comm->allgatherv(send.data(), mine, recv.data(), recvcounts.data())->wait();

    std::size_t at = 0;
    for (int r = 0; r < size; ++r)
        for (std::size_t k = 0; k < recvcounts[uz(r)]; ++k)
            EXPECT_EQ(recv[at++], pairValue(r, 0, k)) << "block " << r << " elem " << k;
}

// Several exchanges in flight at once, waited out of order, with ordinary
// tagged send/recv traffic between the same ranks: the reserved tag and MPI's
// per-pair ordering keep every chunk matched to its own exchange.
TEST_F(MPICommTest, SizeCountExchangesOverlap) {
    const ChunkBytesGuard guard(16);
    const std::size_t per = 5;
    std::vector<std::size_t> counts(uz(size), per);
    std::vector<std::int64_t> sendA(uz(size) * per), sendB(uz(size) * per);
    std::vector<std::int64_t> recvA(uz(size) * per, -1), recvB(uz(size) * per, -1);
    for (int r = 0; r < size; ++r)
        for (std::size_t k = 0; k < per; ++k) {
            sendA[uz(r) * per + k] = pairValue(rank, r, k);
            sendB[uz(r) * per + k] = -pairValue(rank, r, k) - 1;
        }

    auto reqA = comm->alltoallv(sendA.data(), counts.data(), recvA.data(), counts.data());
    auto reqB = comm->alltoallv(sendB.data(), counts.data(), recvB.data(), counts.data());

    // Ring message on the default tag while both exchanges are pending.
    int token_out = rank, token_in = -1;
    auto rreq = comm->recv(&token_in, 1, (rank + size - 1) % size);
    auto sreq = comm->send(&token_out, 1, (rank + 1) % size);

    reqB->wait();
    sreq->wait();
    rreq->wait();
    reqA->wait();

    EXPECT_EQ(token_in, (rank + size - 1) % size);
    for (int s = 0; s < size; ++s)
        for (std::size_t k = 0; k < per; ++k) {
            EXPECT_EQ(recvA[uz(s) * per + k], pairValue(s, rank, k));
            EXPECT_EQ(recvB[uz(s) * per + k], -pairValue(s, rank, k) - 1);
        }
}

TEST_F(MPICommTest, SizeCountSelfMismatchThrows) {
    // Every rank passes the same inconsistent self count, so all throw before
    // any message is issued.
    std::vector<std::size_t> sendcounts(uz(size), 0), recvcounts(uz(size), 0);
    sendcounts[uz(rank)] = 2;
    recvcounts[uz(rank)] = 3;
    std::vector<int> buf(4, 0);
    EXPECT_THROW(comm->alltoallv(buf.data(), sendcounts.data(),
                                 buf.data(), recvcounts.data()),
                 std::invalid_argument);
    EXPECT_THROW(comm->allgatherv(buf.data(), std::size_t{2},
                                  buf.data(), recvcounts.data()),
                 std::invalid_argument);
}

TEST(UtilsTest, DisplacementOverflowThrows) {
    const int counts[3] = {INT_MAX, 1, 1};  // displs[2] = INT_MAX + 1
    EXPECT_THROW(stComm::Utils::calculateDisplacements(counts, 3), std::overflow_error);
    const int ok[3] = {10, 20, 30};
    EXPECT_EQ(stComm::Utils::calculateDisplacements(ok, 3), (std::vector<int>{0, 10, 30}));
}

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
            EXPECT_EQ(recv_data[uz(i)], send_data[uz(i)]);
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
            EXPECT_EQ(recv_data[uz(i)], send_data[uz(i)]);
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
        for (int i = 0; i < N; ++i) buf[uz(i)] = i * 7 + 1;
    }
    auto req = comm->bcast(buf.data(), N, 0);
    req->wait();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(buf[uz(i)], i * 7 + 1) << "rank " << rank << " mismatch at " << i;
    }
}

TEST_F(MPICommTest, BcastDouble) {
    const int N = 64;
    std::vector<double> buf(N);
    if (rank == 0) {
        for (int i = 0; i < N; ++i) buf[uz(i)] = 1.5 * i;
    }
    auto req = comm->bcast(buf.data(), N, 0);
    req->wait();
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(buf[uz(i)], 1.5 * i);
    }
}

TEST_F(MPICommTest, BcastFromNonRootZero) {
    // Broadcast from a non-zero root (only meaningful when size >= 2).
    if (size < 2) GTEST_SKIP();
    const int N = 50;
    const int root = size - 1;
    std::vector<int64_t> buf(N, -1);
    if (rank == root) {
        for (int i = 0; i < N; ++i) buf[uz(i)] = i * 13;
    }
    auto req = comm->bcast(buf.data(), N, root);
    req->wait();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(buf[uz(i)], i * 13);
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
    const uint64_t my_value = uint64_t{100} + static_cast<uint64_t>(rank);
    uint64_t result = comm->exscan<uint64_t>(my_value, MPI_SUM);
    // exclusive prefix: sum_{k=0..r-1}(100 + k) = r*100 + r*(r-1)/2
    const uint64_t expected = static_cast<uint64_t>(rank) * uint64_t{100}
                            + static_cast<uint64_t>(rank) * static_cast<uint64_t>(rank - 1) / 2;
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
