#pragma once

#include "types.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <vector>

namespace stComm {

/**
 * @brief Abstract base class for async operation handles.
 *
 * Two concrete backends inherit this: MPIRequest (host) and NCCLRequest
 * (device). The common path only needs wait()/test(); getBackend() lets a
 * caller holding a base pointer discover the concrete type and downcast to a
 * native sync handle.
 */
class Request {
public:
    virtual ~Request() = default;

    /// @brief Block until the operation completes (runs the finalizer, if any).
    virtual void wait() = 0;

    /// @brief Non-blocking completion test; returns true once complete.
    virtual bool test() = 0;

    /// @brief Current status.
    virtual Status getStatus() const = 0;

    /// @brief Identify the producing backend (safe downcast discriminator).
    virtual Backend getBackend() const = 0;

protected:
    Request() = default;
};

/**
 * @brief MPI request handle.
 *
 * Owns one or more MPI_Request handles — a single one for normal transfers, or
 * several when a >2GB payload is chunked into multiple non-blocking calls
 * (which is why there is no separate "multi" class). wait() drains them all in
 * one MPI_Waitall. May additionally hold scratch storage and a finalizer; async
 * reductions use these to keep internal buffers alive and to deliver the scalar
 * result on completion.
 */
class MPIRequest : public Request {
public:
    MPIRequest() : status_(Status::PENDING) {}

    /// @brief Reserve a fresh MPI_Request slot to pass to a non-blocking call.
    MPI_Request& addHandle() {
        handles_.push_back(MPI_REQUEST_NULL);
        return handles_.back();
    }

    /// @brief Attach an already-issued MPI_Request.
    void addRequest(MPI_Request req) { handles_.push_back(req); }

    /// @brief Keep internal buffers alive until the operation completes.
    void setScratch(std::shared_ptr<void> scratch) { scratch_ = std::move(scratch); }

    /// @brief Run once when the operation completes (e.g. unpack a result).
    void setFinalizer(std::function<void()> fn) { finalizer_ = std::move(fn); }

    void wait() override {
        if (!handles_.empty())
            MPI_Waitall(static_cast<int>(handles_.size()), handles_.data(),
                        MPI_STATUSES_IGNORE);
        complete();
    }

    bool test() override {
        if (!handles_.empty()) {
            int flag;
            MPI_Testall(static_cast<int>(handles_.size()), handles_.data(),
                        &flag, MPI_STATUSES_IGNORE);
            if (!flag) return false;
        }
        complete();
        return true;
    }

    Status getStatus() const override { return status_; }
    Backend getBackend() const override { return Backend::MPI; }

    /// @brief Native handle for advanced callers. Returns the first slot
    /// (creating one if empty); use getHandles() for a chunked request.
    MPI_Request& getHandle() { return handles_.empty() ? addHandle() : handles_.front(); }
    std::vector<MPI_Request>& getHandles() { return handles_; }

private:
    void complete() {
        if (status_ == Status::SUCCESS) return;
        status_ = Status::SUCCESS;
        if (finalizer_) finalizer_();
    }

    std::vector<MPI_Request> handles_;
    std::shared_ptr<void>    scratch_;
    std::function<void()>    finalizer_;
    Status                   status_;
};

/**
 * @brief NCCL request handle (CUDA stream-based).
 *
 * A single CUDA stream serializes and aggregates every enqueued NCCL/CUDA op,
 * so one cudaStreamSynchronize waits for the whole sequence — no "multi" form
 * is needed. scratch/finalizer mirror MPIRequest for stream-based emulated
 * reductions (gather on the stream, reduce on the host once it drains).
 */
class NCCLRequest : public Request {
public:
    explicit NCCLRequest(cudaStream_t stream)
        : stream_(stream), status_(Status::PENDING) {}

    /// @brief Keep internal buffers alive until the operation completes.
    void setScratch(std::shared_ptr<void> scratch) { scratch_ = std::move(scratch); }

    /// @brief Run once when the operation completes (e.g. host-side reduce).
    void setFinalizer(std::function<void()> fn) { finalizer_ = std::move(fn); }

    void wait() override {
        if (stream_ != nullptr) cudaStreamSynchronize(stream_);
        complete();
    }

    bool test() override {
        if (stream_ != nullptr && cudaStreamQuery(stream_) != cudaSuccess) return false;
        complete();
        return true;
    }

    Status getStatus() const override { return status_; }
    Backend getBackend() const override { return Backend::NCCL; }

    cudaStream_t getStream() const { return stream_; }

private:
    void complete() {
        if (status_ == Status::SUCCESS) return;
        status_ = Status::SUCCESS;
        if (finalizer_) finalizer_();
    }

    cudaStream_t          stream_;
    std::shared_ptr<void> scratch_;
    std::function<void()> finalizer_;
    Status                status_;
};

using MPIRequestPtr  = std::shared_ptr<MPIRequest>;
using NCCLRequestPtr = std::shared_ptr<NCCLRequest>;

} // namespace stComm
