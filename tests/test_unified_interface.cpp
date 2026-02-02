/**
 * @file test_unified_interface.cpp
 * @brief Tests for unified Communicator interface
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include <vector>

class UnifiedInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Tests will create their own communicators
    }

    void TearDown() override {
    }
};

// Test MPI backend creation
TEST_F(UnifiedInterfaceTest, CreateMPIBackend) {
    stComm::Communicator comm(stComm::Backend::MPI);

    EXPECT_EQ(comm.getBackend(), stComm::Backend::MPI);
    EXPECT_TRUE(comm.isInitialized());
    EXPECT_GE(comm.getRank(), 0);
    EXPECT_GT(comm.getSize(), 0);
}

// Test send/recv with unified interface
TEST_F(UnifiedInterfaceTest, SendRecvMPI) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    std::vector<int> data(100);

    if (rank == 0) {
        for (int i = 0; i < 100; ++i) {
            data[i] = i;
        }
        auto req = comm.send(data.data(), data.size(), 1);
        req->wait();
    } else if (rank == 1) {
        auto req = comm.recv(data.data(), data.size(), 0);
        req->wait();

        for (int i = 0; i < 100; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }
}

// Test allgather with unified interface
TEST_F(UnifiedInterfaceTest, AllGather) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    std::vector<int> sendbuf(10, rank);
    std::vector<int> recvbuf(10 * size);

    auto req = comm.allgather(sendbuf.data(), 10, recvbuf.data());
    req->wait();

    // Verify data from each rank
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(recvbuf[r * 10 + i], r);
        }
    }
}

// Test allgather_vec with unified interface
TEST_F(UnifiedInterfaceTest, AllGatherVec) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    std::vector<int> sendbuf(5, rank * 10);
    auto recvbuf = comm.allgather_vec(sendbuf);

    ASSERT_EQ(recvbuf.size(), sendbuf.size() * size);

    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < 5; ++i) {
            EXPECT_EQ(recvbuf[r * 5 + i], r * 10);
        }
    }
}

// Test allgatherv_auto with unified interface
TEST_F(UnifiedInterfaceTest, AllGathervAuto) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    // Each rank sends different amount
    int sendcount = rank + 1;
    std::vector<int> sendbuf(sendcount, rank);

    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = i + 1;
    }

    int total = stComm::Utils::total_size(recvcounts);
    std::vector<int> recvbuf(total);

    // Use unified interface with auto displacement
    auto req = comm.allgatherv_auto(sendbuf.data(), sendcount,
                                    recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify data
    auto displs = stComm::Utils::calculate_displacements(recvcounts);
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < recvcounts[r]; ++i) {
            EXPECT_EQ(recvbuf[displs[r] + i], r);
        }
    }
}

// Test allgatherv_vec with unified interface
TEST_F(UnifiedInterfaceTest, AllGathervVec) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    std::vector<int> sendbuf(rank + 1, rank * 100);

    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = i + 1;
    }

    auto recvbuf = comm.allgatherv_vec(sendbuf, recvcounts);

    int expected_size = stComm::Utils::total_size(recvcounts);
    ASSERT_EQ(recvbuf.size(), expected_size);

    auto displs = stComm::Utils::calculate_displacements(recvcounts);
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < recvcounts[r]; ++i) {
            EXPECT_EQ(recvbuf[displs[r] + i], r * 100);
        }
    }
}

// Test alltoallv_auto with unified interface
TEST_F(UnifiedInterfaceTest, AllToAllvAuto) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    std::vector<int> sendcounts(size, 2);
    std::vector<int> recvcounts(size, 2);

    int total_send = stComm::Utils::total_size(sendcounts);
    int total_recv = stComm::Utils::total_size(recvcounts);

    std::vector<int> sendbuf(total_send);
    std::vector<int> recvbuf(total_recv);

    // Fill sendbuf
    for (int i = 0; i < total_send; ++i) {
        sendbuf[i] = rank * 100 + i;
    }

    // Use unified interface
    auto req = comm.alltoallv_auto(sendbuf.data(), sendcounts.data(),
                                   recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify received data
    for (int src = 0; src < size; ++src) {
        for (int i = 0; i < 2; ++i) {
            EXPECT_EQ(recvbuf[src * 2 + i], src * 100 + rank * 2 + i);
        }
    }
}

// Test alltoallv_vec with unified interface
TEST_F(UnifiedInterfaceTest, AllToAllvVec) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    std::vector<int> sendcounts(size, 3);
    std::vector<int> recvcounts(size, 3);

    std::vector<int> sendbuf(size * 3);
    for (size_t i = 0; i < sendbuf.size(); ++i) {
        sendbuf[i] = rank * 1000 + i;
    }

    auto recvbuf = comm.alltoallv_vec(sendbuf, sendcounts, recvcounts);

    ASSERT_EQ(recvbuf.size(), size * 3);

    for (int src = 0; src < size; ++src) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(recvbuf[src * 3 + i], src * 1000 + rank * 3 + i);
        }
    }
}

// Test allreduce with unified interface
TEST_F(UnifiedInterfaceTest, AllReduce) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    std::vector<int> sendbuf(10, rank + 1);
    std::vector<int> recvbuf(10);

    auto req = comm.allreduce(sendbuf.data(), recvbuf.data(), 10);
    req->wait();

    int expected_sum = size * (size + 1) / 2;
    for (auto val : recvbuf) {
        EXPECT_EQ(val, expected_sum);
    }
}

// Test barrier with unified interface
TEST_F(UnifiedInterfaceTest, Barrier) {
    stComm::Communicator comm(stComm::Backend::MPI);

    // Should not throw
    EXPECT_NO_THROW(comm.barrier());
}

// Test backend query
TEST_F(UnifiedInterfaceTest, BackendQuery) {
    stComm::Communicator comm(stComm::Backend::MPI);

    EXPECT_EQ(comm.getBackend(), stComm::Backend::MPI);
    EXPECT_NE(comm.getMPIComm(), nullptr);
    EXPECT_EQ(comm.getNCCLComm(), nullptr);
}

// Test different data types
TEST_F(UnifiedInterfaceTest, DifferentTypes) {
    stComm::Communicator comm(stComm::Backend::MPI);

    int rank = comm.getRank();
    int size = comm.getSize();

    // Test with double
    {
        std::vector<double> sendbuf(5, rank * 1.5);
        auto recvbuf = comm.allgather_vec(sendbuf);

        ASSERT_EQ(recvbuf.size(), sendbuf.size() * size);

        for (int r = 0; r < size; ++r) {
            for (int i = 0; i < 5; ++i) {
                EXPECT_NEAR(recvbuf[r * 5 + i], r * 1.5, 1e-9);
            }
        }
    }

    // Test with float
    {
        std::vector<float> sendbuf(3, rank * 2.5f);
        std::vector<float> recvbuf(3 * size);

        auto req = comm.allgather(sendbuf.data(), 3, recvbuf.data());
        req->wait();

        for (int r = 0; r < size; ++r) {
            for (int i = 0; i < 3; ++i) {
                EXPECT_NEAR(recvbuf[r * 3 + i], r * 2.5f, 1e-6f);
            }
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    stComm::MPIComm::initialize(&argc, &argv);

    int result = RUN_ALL_TESTS();

    stComm::MPIComm::finalize();
    return result;
}
