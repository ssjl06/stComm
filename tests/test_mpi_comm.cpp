/**
 * @file test_mpi_comm.cpp
 * @brief Tests for MPIComm class
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include <vector>

class MPICommTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm = std::make_unique<stComm::MPIComm>();
        rank = comm->getRank();
        size = comm->getSize();
    }

    void TearDown() override {
        comm.reset();
    }

    std::unique_ptr<stComm::MPIComm> comm;
    int rank;
    int size;
};

// Test basic properties
TEST_F(MPICommTest, BasicProperties) {
    EXPECT_GE(rank, 0);
    EXPECT_LT(rank, size);
    EXPECT_GT(size, 0);
    EXPECT_EQ(comm->getBackend(), stComm::Backend::MPI);
}

// Test barrier
TEST_F(MPICommTest, Barrier) {
    EXPECT_NO_THROW(comm->barrier());
}

// Test send/recv
TEST_F(MPICommTest, SendRecv) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    std::vector<int> data(100);

    if (rank == 0) {
        for (int i = 0; i < 100; ++i) {
            data[i] = i;
        }
        auto req = comm->send(data.data(), data.size(), 1, 0);
        req->wait();
    } else if (rank == 1) {
        auto req = comm->recv(data.data(), data.size(), 0, 0);
        req->wait();

        for (int i = 0; i < 100; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }
}

// Test allgatherv with auto displacement
TEST_F(MPICommTest, Allgatherv) {
    // Each rank sends different amount
    int sendcount = rank + 1;
    std::vector<int> sendbuf(sendcount);
    for (int i = 0; i < sendcount; ++i) {
        sendbuf[i] = rank * 100 + i;
    }

    // Prepare receive counts
    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = i + 1;
    }

    // Calculate total receive size
    int total_recv = stComm::Utils::totalSize(recvcounts);
    std::vector<int> recvbuf(total_recv);

    // Call allgatherv (displacement auto-calculated)
    auto req = comm->allgatherv(sendbuf.data(), sendcount, recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify data
    auto displs = stComm::Utils::calculateDisplacements(recvcounts);
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < recvcounts[r]; ++i) {
            EXPECT_EQ(recvbuf[displs[r] + i], r * 100 + i);
        }
    }
}

// Test alltoallv with auto displacement
TEST_F(MPICommTest, Alltoallv) {
    // Each rank sends 2 items to each other rank
    std::vector<int> sendcounts(size, 2);
    std::vector<int> recvcounts(size, 2);

    int total_send = stComm::Utils::totalSize(sendcounts);
    int total_recv = stComm::Utils::totalSize(recvcounts);

    std::vector<int> sendbuf(total_send);
    std::vector<int> recvbuf(total_recv);

    // Fill sendbuf
    auto send_displs = stComm::Utils::calculateDisplacements(sendcounts);
    for (int dest = 0; dest < size; ++dest) {
        for (int i = 0; i < sendcounts[dest]; ++i) {
            sendbuf[send_displs[dest] + i] = rank * 1000 + dest * 100 + i;
        }
    }

    // Call alltoallv (displacement auto-calculated)
    auto req = comm->alltoallv(sendbuf.data(), sendcounts.data(),
                               recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify received data
    auto recv_displs = stComm::Utils::calculateDisplacements(recvcounts);
    for (int src = 0; src < size; ++src) {
        for (int i = 0; i < recvcounts[src]; ++i) {
            int expected = src * 1000 + rank * 100 + i;
            EXPECT_EQ(recvbuf[recv_displs[src] + i], expected);
        }
    }
}

// Test with different data types
TEST_F(MPICommTest, DifferentTypes) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // Test with double
    std::vector<double> double_data(50);

    if (rank == 0) {
        for (int i = 0; i < 50; ++i) {
            double_data[i] = i * 1.5;
        }
        auto req = comm->send(double_data.data(), double_data.size(), 1, 0);
        req->wait();
    } else if (rank == 1) {
        auto req = comm->recv(double_data.data(), double_data.size(), 0, 0);
        req->wait();

        for (int i = 0; i < 50; ++i) {
            EXPECT_NEAR(double_data[i], i * 1.5, 1e-9);
        }
    }
}

// Test async operations
TEST_F(MPICommTest, AsyncOperations) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    std::vector<int> data(100, rank);

    if (rank == 0) {
        auto req = comm->send(data.data(), data.size(), 1, 0);

        // Do other work while send is in progress
        int dummy = 0;
        for (int i = 0; i < 1000; ++i) {
            dummy += i;
        }

        // Wait for completion
        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);
    } else if (rank == 1) {
        auto req = comm->recv(data.data(), data.size(), 0, 0);

        // Test operation
        EXPECT_FALSE(req->test()); // Might still be pending

        // Wait for completion
        req->wait();
        EXPECT_EQ(req->getStatus(), stComm::Status::SUCCESS);

        for (auto val : data) {
            EXPECT_EQ(val, 0);
        }
    }
}
