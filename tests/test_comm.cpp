/**
 * @file test_comm.cpp
 * @brief Tests for the unified stComm::Comm facade.
 *
 * Host-space collectives run anywhere MPI does (no GPU needed). Device-space
 * collectives are guarded: they SKIP unless every rank gets its own GPU, but
 * still compile so the Device template paths are exercised on a 2+ GPU box.
 *
 * Expected results are computed analytically rather than against a second
 * backend object, so two non-blocking collectives are never in flight on the
 * same communicator at once.
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include <cuda_runtime.h>
#include <vector>
#include <type_traits>

using stComm::Comm;
using stComm::Space;
using stComm::ReduceOp;
using stComm::Backend;
using stComm::NCCLRequest;

class CommFacadeTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        cudaGetDeviceCount(&num_gpus);
    }

    bool deviceUsable() const { return num_gpus >= size; }

    int rank = 0;
    int size = 0;
    int num_gpus = 0;
};

// ============================================================================
// Host-space facade — runs without a GPU
// ============================================================================

TEST_F(CommFacadeTest, RankSizeAndNoDevice) {
    Comm comm;  // host-only
    EXPECT_EQ(comm.getRank(), rank);
    EXPECT_EQ(comm.getSize(), size);
    EXPECT_FALSE(comm.hasDevice());
    EXPECT_EQ(comm.mpi().getRank(), rank);  // escape hatch agrees
}

TEST_F(CommFacadeTest, BcastHost) {
    Comm comm;
    const int root = 0;
    const int n = 5;
    std::vector<int> buf(n, -1);
    if (rank == root)
        for (int i = 0; i < n; ++i) buf[i] = 1000 + i;

    auto req = comm.bcast<Space::Host>(buf.data(), n, root);
    // Host return is the concrete MPIRequest, symmetric with the device side.
    static_assert(std::is_same_v<decltype(req), stComm::MPIRequestPtr>);
    EXPECT_EQ(req->getBackend(), Backend::MPI);
    req->wait();

    for (int i = 0; i < n; ++i) EXPECT_EQ(buf[i], 1000 + i);
}

TEST_F(CommFacadeTest, AllgathervHost) {
    Comm comm;
    const int count = 3;
    std::vector<int> send(count);
    for (int k = 0; k < count; ++k) send[k] = rank * 10 + k;

    std::vector<int> recvcounts(size, count);
    std::vector<int> recv(size * count, -1);

    comm.allgatherv<Space::Host>(send.data(), count, recv.data(), recvcounts.data())->wait();

    for (int i = 0; i < size; ++i)
        for (int k = 0; k < count; ++k)
            EXPECT_EQ(recv[i * count + k], i * 10 + k) << "block " << i << " elem " << k;
}

TEST_F(CommFacadeTest, AlltoallvHost) {
    Comm comm;
    const int per = 2;
    std::vector<int> sendcounts(size, per), recvcounts(size, per);
    std::vector<int> send(size * per), recv(size * per, -1);
    // Block destined for peer j encodes (this rank -> j).
    for (int j = 0; j < size; ++j)
        for (int k = 0; k < per; ++k)
            send[j * per + k] = rank * 100 + j * 10 + k;

    comm.alltoallv<Space::Host>(send.data(), sendcounts.data(),
                                recv.data(), recvcounts.data())->wait();

    // Block received from peer i was the block i destined for this rank.
    for (int i = 0; i < size; ++i)
        for (int k = 0; k < per; ++k)
            EXPECT_EQ(recv[i * per + k], i * 100 + rank * 10 + k) << "from " << i << " elem " << k;
}

TEST_F(CommFacadeTest, AllreduceMaxlocHost) {
    Comm comm;
    // Async: result lands in the out-param once the request completes.
    // value == rank → max lives on the highest rank, no ties.
    std::pair<double, int> hi;
    comm.allreduceMaxloc<Space::Host>(static_cast<double>(rank), &hi)->wait();
    EXPECT_DOUBLE_EQ(hi.first, static_cast<double>(size - 1));
    EXPECT_EQ(hi.second, size - 1);

    // value == -rank → max lives on rank 0.
    std::pair<double, int> lo;
    comm.allreduceMaxloc<Space::Host>(-static_cast<double>(rank), &lo)->wait();
    EXPECT_DOUBLE_EQ(lo.first, 0.0);
    EXPECT_EQ(lo.second, 0);
}

TEST_F(CommFacadeTest, ExscanReduceOpHost) {
    Comm comm;
    const int v = rank + 1;  // ranks contribute 1, 2, 3, ...

    // Sum: exclusive prefix sum of (k+1) for k < rank; rank 0 → identity 0.
    int expSum = 0;
    for (int k = 0; k < rank; ++k) expSum += (k + 1);
    int gotSum = -1;
    comm.exscan<Space::Host>(v, &gotSum, ReduceOp::Sum)->wait();
    EXPECT_EQ(gotSum, expSum);

    // Max: prefix max of (k+1); rank 0 → identity 0 (MPI_Exscan leaves it unset,
    // MPIComm returns T{}).
    int expMax = (rank == 0) ? 0 : rank;  // max of {1..rank} == rank
    int gotMax = -1;
    comm.exscan<Space::Host>(v, &gotMax, ReduceOp::Max)->wait();
    EXPECT_EQ(gotMax, expMax);
}

// ============================================================================
// Device-space facade — SKIPs unless each rank owns a GPU
// ============================================================================

TEST_F(CommFacadeTest, BcastDevice) {
    if (!deviceUsable())
        GTEST_SKIP() << "needs " << size << " GPUs, have " << num_gpus;

    Comm comm = Comm::onDevice(rank);  // device_id == rank
    ASSERT_TRUE(comm.hasDevice());
    EXPECT_EQ(comm.nccl().getRank(), rank);  // escape hatch agrees

    cudaSetDevice(rank);
    const int root = 0;
    const int n = 5;
    std::vector<int> host(n, -1);
    if (rank == root)
        for (int i = 0; i < n; ++i) host[i] = 2000 + i;

    int* dbuf = nullptr;
    cudaMalloc(&dbuf, n * sizeof(int));
    cudaMemcpy(dbuf, host.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    auto req = comm.bcast<Space::Device>(dbuf, n, root);
    // Device return is the concrete NCCLRequest — getStream() with no cast.
    static_assert(std::is_same_v<decltype(req), stComm::NCCLRequestPtr>);
    EXPECT_EQ(req->getBackend(), Backend::NCCL);
    EXPECT_NE(req->getStream(), nullptr);
    req->wait();

    cudaMemcpy(host.data(), dbuf, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_EQ(host[i], 2000 + i);

    cudaFree(dbuf);
}

TEST_F(CommFacadeTest, ReductionsDevice) {
    if (!deviceUsable())
        GTEST_SKIP() << "needs " << size << " GPUs, have " << num_gpus;

    Comm comm = Comm::onDevice(rank);
    cudaSetDevice(rank);

    // Emulated MAXLOC: value == rank → max on the highest rank. Result is
    // delivered to the host out-param once the (stream-backed) request drains.
    std::pair<double, int> hi;
    comm.allreduceMaxloc<Space::Device>(static_cast<double>(rank), &hi)->wait();
    EXPECT_DOUBLE_EQ(hi.first, static_cast<double>(size - 1));
    EXPECT_EQ(hi.second, size - 1);

    // Emulated exclusive scan (Sum): prefix sum of (k+1) for k < rank.
    int expSum = 0;
    for (int k = 0; k < rank; ++k) expSum += (k + 1);
    int gotSum = -1;
    comm.exscan<Space::Device>(rank + 1, &gotSum, ReduceOp::Sum)->wait();
    EXPECT_EQ(gotSum, expSum);
}
