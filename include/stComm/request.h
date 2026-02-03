#pragma once

#include "types.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>

namespace stComm {

/**
 * @brief Abstract base class for async operation handles
 */
class Request {
public:
    virtual ~Request() = default;

    /**
     * @brief Wait for operation to complete
     */
    virtual void wait() = 0;

    /**
     * @brief Test if operation is complete (non-blocking)
     * @return true if complete, false otherwise
     */
    virtual bool test() = 0;

    /**
     * @brief Get operation status
     */
    virtual Status getStatus() const = 0;

protected:
    Request() = default;
};

/**
 * @brief MPI request handle
 */
class MPIRequest : public Request {
public:
    MPIRequest() : req_(MPI_REQUEST_NULL), status_(Status::PENDING) {}

    void wait() override {
        if (req_ != MPI_REQUEST_NULL) {
            MPI_Wait(&req_, MPI_STATUS_IGNORE);
            status_ = Status::SUCCESS;
        }
    }

    bool test() override {
        if (req_ == MPI_REQUEST_NULL) {
            return true;
        }
        int flag;
        MPI_Test(&req_, &flag, MPI_STATUS_IGNORE);
        if (flag) {
            status_ = Status::SUCCESS;
            return true;
        }
        return false;
    }

    Status getStatus() const override {
        return status_;
    }

    MPI_Request& getHandle() { return req_; }

private:
    MPI_Request req_;
    Status status_;
};

/**
 * @brief Multi MPI request handle for chunked large data transfers
 * Manages multiple MPI_Request objects for >2GB data transfers
 */
class MultiMPIRequest : public Request {
public:
    MultiMPIRequest() : status_(Status::PENDING) {}

    void addRequest(MPI_Request req) {
        requests_.push_back(req);
    }

    void wait() override {
        if (!requests_.empty()) {
            MPI_Waitall(requests_.size(), requests_.data(), MPI_STATUSES_IGNORE);
            status_ = Status::SUCCESS;
        }
    }

    bool test() override {
        if (requests_.empty()) {
            return true;
        }
        int flag;
        MPI_Testall(requests_.size(), requests_.data(), &flag, MPI_STATUSES_IGNORE);
        if (flag) {
            status_ = Status::SUCCESS;
            return true;
        }
        return false;
    }

    Status getStatus() const override {
        return status_;
    }

private:
    std::vector<MPI_Request> requests_;
    Status status_;
};

/**
 * @brief NCCL request handle (CUDA stream-based)
 */
class NCCLRequest : public Request {
public:
    explicit NCCLRequest(cudaStream_t stream)
        : stream_(stream), status_(Status::PENDING) {}

    void wait() override {
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
            status_ = Status::SUCCESS;
        }
    }

    bool test() override {
        if (stream_ == nullptr) {
            return true;
        }
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess) {
            status_ = Status::SUCCESS;
            return true;
        }
        return false;
    }

    Status getStatus() const override {
        return status_;
    }

    cudaStream_t getStream() const { return stream_; }

private:
    cudaStream_t stream_;
    Status status_;
};

} // namespace stComm
