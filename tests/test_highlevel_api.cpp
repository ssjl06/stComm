/**
 * @file test_highlevel_api.cpp
 * @brief Tests for high-level convenience APIs
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"
#include <vector>
#include <numeric>

class HighLevelAPITest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<stComm::MPIComm>();
        rank_ = comm_->getRank();
        size_ = comm_->getSize();
    }

    void TearDown() override {
        comm_.reset();
    }

    std::unique_ptr<stComm::MPIComm> comm_;
    int rank_;
    int size_;
};

// Test Utils class
TEST_F(HighLevelAPITest, UtilsCalculateDisplacements) {
    std::vector<int> counts = {10, 20, 30, 40};
    auto displs = stComm::Utils::calculate_displacements(counts);

    ASSERT_EQ(displs.size(), counts.size());
    EXPECT_EQ(displs[0], 0);
    EXPECT_EQ(displs[1], 10);
    EXPECT_EQ(displs[2], 30);
    EXPECT_EQ(displs[3], 60);
}

TEST_F(HighLevelAPITest, UtilsTotalSize) {
    std::vector<int> counts = {10, 20, 30, 40};
    int total = stComm::Utils::total_size(counts);
    EXPECT_EQ(total, 100);
}

TEST_F(HighLevelAPITest, CommBuffer) {
    std::vector<int> counts = {10, 20, 30};
    stComm::CommBuffer<int> buffer(counts);

    EXPECT_EQ(buffer.size(), 60);
    EXPECT_NE(buffer.data(), nullptr);

    // Test write/read
    buffer[0] = 42;
    EXPECT_EQ(buffer[0], 42);
}

// Test allgather_vec
TEST_F(HighLevelAPITest, AllGatherVec) {
    std::vector<int> sendbuf(5);
    for (int i = 0; i < 5; ++i) {
        sendbuf[i] = rank_ * 10 + i;
    }

    auto recvbuf = comm_->allgather_vec(sendbuf);

    ASSERT_EQ(recvbuf.size(), sendbuf.size() * size_);

    // Verify data from each rank
    for (int r = 0; r < size_; ++r) {
        for (int i = 0; i < 5; ++i) {
            EXPECT_EQ(recvbuf[r * 5 + i], r * 10 + i);
        }
    }
}

// Test allgatherv_auto
TEST_F(HighLevelAPITest, AllGathervAuto) {
    // Each rank sends different amount
    int sendcount = rank_ + 1;
    std::vector<int> sendbuf(sendcount);
    for (int i = 0; i < sendcount; ++i) {
        sendbuf[i] = rank_ * 100 + i;
    }

    // Prepare receive counts
    std::vector<int> recvcounts(size_);
    for (int i = 0; i < size_; ++i) {
        recvcounts[i] = i + 1;
    }

    int total_recv = stComm::Utils::total_size(recvcounts);
    std::vector<int> recvbuf(total_recv);

    // Use auto displacement version
    auto req = comm_->allgatherv_auto(sendbuf.data(), sendcount, recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify data
    auto displs = stComm::Utils::calculate_displacements(recvcounts);
    for (int r = 0; r < size_; ++r) {
        for (int i = 0; i < recvcounts[r]; ++i) {
            EXPECT_EQ(recvbuf[displs[r] + i], r * 100 + i);
        }
    }
}

// Test allgatherv_vec
TEST_F(HighLevelAPITest, AllGathervVec) {
    std::vector<int> sendbuf(rank_ + 1);
    for (size_t i = 0; i < sendbuf.size(); ++i) {
        sendbuf[i] = rank_ * 100 + i;
    }

    std::vector<int> recvcounts(size_);
    for (int i = 0; i < size_; ++i) {
        recvcounts[i] = i + 1;
    }

    auto recvbuf = comm_->allgatherv_vec(sendbuf, recvcounts);

    // Verify size
    int expected_size = stComm::Utils::total_size(recvcounts);
    ASSERT_EQ(recvbuf.size(), expected_size);

    // Verify data
    auto displs = stComm::Utils::calculate_displacements(recvcounts);
    for (int r = 0; r < size_; ++r) {
        for (int i = 0; i < recvcounts[r]; ++i) {
            EXPECT_EQ(recvbuf[displs[r] + i], r * 100 + i);
        }
    }
}

// Test alltoallv_auto
TEST_F(HighLevelAPITest, AllToAllvAuto) {
    // Each rank sends (rank+1) items to each other rank
    std::vector<int> sendcounts(size_, rank_ + 1);
    std::vector<int> recvcounts(size_);
    for (int i = 0; i < size_; ++i) {
        recvcounts[i] = i + 1;
    }

    int total_send = stComm::Utils::total_size(sendcounts);
    int total_recv = stComm::Utils::total_size(recvcounts);

    std::vector<int> sendbuf(total_send);
    std::vector<int> recvbuf(total_recv);

    // Fill sendbuf with unique data for each destination
    auto send_displs = stComm::Utils::calculate_displacements(sendcounts);
    for (int dest = 0; dest < size_; ++dest) {
        for (int i = 0; i < sendcounts[dest]; ++i) {
            sendbuf[send_displs[dest] + i] = rank_ * 1000 + dest * 100 + i;
        }
    }

    // Use auto displacement version
    auto req = comm_->alltoallv_auto(sendbuf.data(), sendcounts.data(),
                                     recvbuf.data(), recvcounts.data());
    req->wait();

    // Verify received data
    auto recv_displs = stComm::Utils::calculate_displacements(recvcounts);
    for (int src = 0; src < size_; ++src) {
        for (int i = 0; i < recvcounts[src]; ++i) {
            int expected = src * 1000 + rank_ * 100 + i;
            EXPECT_EQ(recvbuf[recv_displs[src] + i], expected);
        }
    }
}

// Test alltoallv_vec
TEST_F(HighLevelAPITest, AllToAllvVec) {
    std::vector<int> sendcounts(size_, 2);  // Send 2 items to each rank
    std::vector<int> recvcounts(size_, 2);  // Receive 2 items from each rank

    std::vector<int> sendbuf(size_ * 2);
    for (size_t i = 0; i < sendbuf.size(); ++i) {
        sendbuf[i] = rank_ * 100 + i;
    }

    auto recvbuf = comm_->alltoallv_vec(sendbuf, sendcounts, recvcounts);

    ASSERT_EQ(recvbuf.size(), size_ * 2);

    // Verify data from each source
    for (int src = 0; src < size_; ++src) {
        for (int i = 0; i < 2; ++i) {
            EXPECT_EQ(recvbuf[src * 2 + i], src * 100 + rank_ * 2 + i);
        }
    }
}

// Test allreduce
TEST_F(HighLevelAPITest, AllReduce) {
    std::vector<int> sendbuf(10, rank_ + 1);  // Each rank contributes (rank+1)
    std::vector<int> recvbuf(10);

    auto req = comm_->allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), MPI_SUM);
    req->wait();

    // Sum should be 1 + 2 + ... + size_ = size_ * (size_ + 1) / 2
    int expected_sum = size_ * (size_ + 1) / 2;
    for (auto val : recvbuf) {
        EXPECT_EQ(val, expected_sum);
    }
}

// Test with double precision
TEST_F(HighLevelAPITest, AllGatherVecDouble) {
    std::vector<double> sendbuf(3);
    for (int i = 0; i < 3; ++i) {
        sendbuf[i] = rank_ * 10.5 + i * 0.1;
    }

    auto recvbuf = comm_->allgather_vec(sendbuf);

    ASSERT_EQ(recvbuf.size(), sendbuf.size() * size_);

    for (int r = 0; r < size_; ++r) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(recvbuf[r * 3 + i], r * 10.5 + i * 0.1, 1e-9);
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
