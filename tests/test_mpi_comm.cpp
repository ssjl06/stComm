#include <gtest/gtest.h>
#include "stComm/mpi_comm.h"
#include <vector>
#include <numeric>

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

TEST_F(MPICommTest, BasicInfo) {
    EXPECT_GE(rank, 0);
    EXPECT_LT(rank, size);
    EXPECT_GT(size, 0);
}

TEST_F(MPICommTest, SendRecvInt) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI processes";
    }

    const int count = 100;
    std::vector<int> send_data(count);
    std::vector<int> recv_data(count, 0);

    // Initialize send data
    std::iota(send_data.begin(), send_data.end(), rank * 1000);

    stComm::RequestPtr send_req, recv_req;

    if (rank == 0) {
        // Rank 0 sends to rank 1
        send_req = comm->send(send_data.data(), count, 1, 0);
    } else if (rank == 1) {
        // Rank 1 receives from rank 0
        recv_req = comm->recv(recv_data.data(), count, 0, 0);
    }

    // Wait for completion
    if (send_req) {
        EXPECT_EQ(send_req->wait(), stComm::Status::SUCCESS);
    }
    if (recv_req) {
        EXPECT_EQ(recv_req->wait(), stComm::Status::SUCCESS);
    }

    // Verify received data
    if (rank == 1) {
        for (int i = 0; i < count; ++i) {
            EXPECT_EQ(recv_data[i], i);
        }
    }
}

TEST_F(MPICommTest, SendRecvDouble) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI processes";
    }

    const int count = 100;
    std::vector<double> send_data(count);
    std::vector<double> recv_data(count, 0.0);

    // Initialize send data
    for (int i = 0; i < count; ++i) {
        send_data[i] = rank * 1000.0 + i * 0.5;
    }

    stComm::RequestPtr send_req, recv_req;

    if (rank == 0) {
        send_req = comm->send(send_data.data(), count, 1, 0);
    } else if (rank == 1) {
        recv_req = comm->recv(recv_data.data(), count, 0, 0);
    }

    if (send_req) {
        EXPECT_EQ(send_req->wait(), stComm::Status::SUCCESS);
    }
    if (recv_req) {
        EXPECT_EQ(recv_req->wait(), stComm::Status::SUCCESS);
    }

    if (rank == 1) {
        for (int i = 0; i < count; ++i) {
            EXPECT_DOUBLE_EQ(recv_data[i], i * 0.5);
        }
    }
}

TEST_F(MPICommTest, AllgathervInt) {
    const int local_count = 10;
    std::vector<int> send_data(local_count);
    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);

    // Each rank has different amount of data
    for (int i = 0; i < size; ++i) {
        recv_counts[i] = local_count + i;
        displs[i] = (i == 0) ? 0 : (displs[i-1] + recv_counts[i-1]);
    }

    int total_count = displs[size-1] + recv_counts[size-1];
    std::vector<int> recv_data(total_count, 0);

    // Initialize send data
    for (int i = 0; i < local_count + rank; ++i) {
        if (i < send_data.size()) {
            send_data[i] = rank * 100 + i;
        }
    }

    // Perform allgatherv
    auto req = comm->allgatherv(send_data.data(), local_count + rank,
                                recv_data.data(), recv_counts.data(), displs.data());

    ASSERT_NE(req, nullptr);
    EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);

    // Verify data from each rank
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < recv_counts[r]; ++i) {
            EXPECT_EQ(recv_data[displs[r] + i], r * 100 + i);
        }
    }
}

TEST_F(MPICommTest, AlltoallvInt) {
    // Each rank sends different amounts to each other rank
    std::vector<int> send_counts(size);
    std::vector<int> send_displs(size);
    std::vector<int> recv_counts(size);
    std::vector<int> recv_displs(size);

    for (int i = 0; i < size; ++i) {
        send_counts[i] = i + 1;
        recv_counts[i] = rank + 1;
        send_displs[i] = (i == 0) ? 0 : (send_displs[i-1] + send_counts[i-1]);
        recv_displs[i] = (i == 0) ? 0 : (recv_displs[i-1] + recv_counts[i-1]);
    }

    int total_send = send_displs[size-1] + send_counts[size-1];
    int total_recv = recv_displs[size-1] + recv_counts[size-1];

    std::vector<int> send_data(total_send);
    std::vector<int> recv_data(total_recv, 0);

    // Initialize send data
    for (int i = 0; i < total_send; ++i) {
        send_data[i] = rank * 1000 + i;
    }

    // Perform alltoallv
    auto req = comm->alltoallv(send_data.data(), send_counts.data(), send_displs.data(),
                               recv_data.data(), recv_counts.data(), recv_displs.data());

    ASSERT_NE(req, nullptr);
    EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);

    // Basic verification - check that we received data
    bool has_data = false;
    for (int i = 0; i < total_recv; ++i) {
        if (recv_data[i] != 0) {
            has_data = true;
            break;
        }
    }
    EXPECT_TRUE(has_data || total_recv == 0);
}

TEST_F(MPICommTest, AsyncTest) {
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI processes";
    }

    const int count = 50;
    std::vector<int> send_data(count);
    std::vector<int> recv_data(count, 0);

    std::iota(send_data.begin(), send_data.end(), 0);

    stComm::RequestPtr req;

    if (rank == 0) {
        req = comm->send(send_data.data(), count, 1, 0);
    } else if (rank == 1) {
        req = comm->recv(recv_data.data(), count, 0, 0);
    }

    if (req) {
        // Test non-blocking check
        bool completed = false;
        stComm::Status status;

        // Poll a few times
        for (int i = 0; i < 5; ++i) {
            status = req->test(completed);
            if (completed) {
                break;
            }
        }

        // Eventually wait
        EXPECT_EQ(req->wait(), stComm::Status::SUCCESS);
    }
}

TEST_F(MPICommTest, Barrier) {
    // Test barrier - all ranks should synchronize
    comm->barrier();

    // If we reach here, barrier worked
    SUCCEED();
}
