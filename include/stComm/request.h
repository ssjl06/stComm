#pragma once

#include "types.h"
#include "cuda_check.h"
#include "mpi_check.h"
#include "nccl_check.h"
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
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
            STCOMM_MPI_CHECK(MPI_Waitall(static_cast<int>(handles_.size()),
                                         handles_.data(), MPI_STATUSES_IGNORE));
        complete();
    }

    bool test() override {
        if (!handles_.empty()) {
            int flag;
            STCOMM_MPI_CHECK(MPI_Testall(static_cast<int>(handles_.size()),
                                         handles_.data(), &flag,
                                         MPI_STATUSES_IGNORE));
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

namespace detail {

// Shared by an NCCLComm and every request it issues, so whichever wait first
// sees the communicator fail aborts it exactly once, and ~NCCLComm then skips
// ncclCommDestroy on the dead handle. Once `aborted` is set, `comm` is a
// destroyed handle (nulled right after the abort) and every pending or later
// wait/test/submission must fail instead of touching it — aborted work may
// even look "complete" to its event. `comm` is also nulled when the NCCLComm
// is destroyed so a request that outlives it stops polling the handle.
struct NCCLCommState {
    ncclComm_t        comm = nullptr;
    std::atomic<bool> aborted{false};
};

// STCOMM_NCCL_TIMEOUT_SEC: optional wall-clock limit for one NCCL wait, read
// once. Unset or <= 0 means no limit (the default — a legitimately long
// collective must never be cut off unless the user asks for it).
inline double ncclWaitTimeoutSec() {
    static const double sec = [] {
        const char* s = std::getenv("STCOMM_NCCL_TIMEOUT_SEC");
        return s ? std::atof(s) : 0.0;
    }();
    return sec;
}

[[noreturn]] inline void abortAndThrow(NCCLCommState& state, const std::string& msg) {
    if (!state.aborted.exchange(true)) {
        ncclCommAbort(state.comm);
        state.comm = nullptr;  // destroyed by the abort — never query it again
    }
    throw std::runtime_error("stComm NCCL: " + msg);
}

inline void throwIfAborted(const NCCLCommState* state, const char* what) {
    if (state != nullptr && state->aborted.load()) {
        throw std::runtime_error(std::string("stComm NCCL: ") + what +
                                 ": communicator was aborted after an earlier failure");
    }
}

// Wait until `query()` (cudaEventQuery / cudaStreamQuery) stops reporting
// cudaErrorNotReady. Busy-polls like cudaEventSynchronize's default spin, but
// every kAsyncCheckInterval polls also asks NCCL whether the communicator has
// failed (network fault, dead peer): the event would then never fire and a
// plain synchronize would hang until the job's walltime. Short ops complete
// before the first check and pay nothing extra. On an async error or timeout,
// abort the communicator and throw.
template<typename Query>
void ncclPollWait(Query query, NCCLCommState* state, const char* what) {
    constexpr std::uint64_t kAsyncCheckInterval = 1024;
    std::chrono::steady_clock::time_point start{};
    for (std::uint64_t n = 1;; ++n) {
        // Checked before the query: work on an aborted communicator can let
        // its event fire, which must not be mistaken for success.
        throwIfAborted(state, what);
        const cudaError_t err = query();
        if (err == cudaSuccess) return;
        if (err != cudaErrorNotReady) STCOMM_CUDA_CHECK(err);
        if (n % kAsyncCheckInterval != 0 || state == nullptr || state->comm == nullptr) {
            continue;
        }

        ncclResult_t async_err = ncclSuccess;
        STCOMM_NCCL_CHECK(ncclCommGetAsyncError(state->comm, &async_err));
        if (async_err != ncclSuccess && async_err != ncclInProgress) {
            abortAndThrow(*state, std::string(what) + ": communicator failed: " +
                                  ncclGetErrorString(async_err));
        }

        const double limit = ncclWaitTimeoutSec();
        if (limit > 0) {
            // The clock starts at the first check, a few µs in — negligible
            // against any limit worth setting.
            const auto now = std::chrono::steady_clock::now();
            if (n == kAsyncCheckInterval) {
                start = now;
            } else if (std::chrono::duration<double>(now - start).count() > limit) {
                abortAndThrow(*state, std::string(what) + ": no completion within " +
                                      std::to_string(limit) + " s (STCOMM_NCCL_TIMEOUT_SEC)");
            }
        }
    }
}

}  // namespace detail

/**
 * @brief NCCL request handle (CUDA event-based).
 *
 * NCCLComm enqueues every op on its single internal stream (NCCL requires a
 * consistent issue order per communicator, so one stream keeps that ordering
 * trivially correct). To make completion *per request* rather than per stream,
 * each request records its own cudaEvent right after its ops are enqueued;
 * wait()/test() then track that event, so r1.wait() does not block on a later
 * r2's work that happens to share the stream. scratch/finalizer mirror
 * MPIRequest for the emulated reductions (gather on the stream, reduce on the
 * host once the event signals).
 *
 * Lifecycle: NCCLComm constructs the request, enqueues the ops, then calls
 * record(stream, state). Until record() runs the request is "not yet recorded" and
 * wait()/test() complete immediately — this is what lets a request issued
 * inside a user groupStart()/groupEnd() defer its event to groupEnd(), where
 * the ops are actually flushed to the stream.
 */
class NCCLRequest : public Request {
public:
    NCCLRequest() : status_(Status::PENDING) {
        // Timing is never needed; disabling it makes the event cheaper.
        STCOMM_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }

    ~NCCLRequest() override {
        // Destructors must not throw, so swallow any error here.
        if (event_ != nullptr) cudaEventDestroy(event_);
    }

    NCCLRequest(const NCCLRequest&)            = delete;
    NCCLRequest& operator=(const NCCLRequest&) = delete;

    /// @brief Mark this request's completion point on `stream`. Called by
    /// NCCLComm once the request's ops have been enqueued (after groupEnd() for
    /// grouped ops). wait()/test() are no-ops until this runs. `state` lets
    /// wait()/test() detect (and abort on) an asynchronous communicator failure.
    void record(cudaStream_t stream, std::shared_ptr<detail::NCCLCommState> state) {
        stream_ = stream;
        state_  = std::move(state);
        STCOMM_CUDA_CHECK(cudaEventRecord(event_, stream));
        recorded_ = true;
    }

    /// @brief Keep internal buffers alive until the operation completes.
    void setScratch(std::shared_ptr<void> scratch) { scratch_ = std::move(scratch); }

    /// @brief Run once when the operation completes (e.g. host-side reduce).
    void setFinalizer(std::function<void()> fn) { finalizer_ = std::move(fn); }

    void wait() override {
        if (status_ == Status::SUCCESS) return;  // completed before any abort
        if (recorded_) {
            detail::ncclPollWait([this] { return cudaEventQuery(event_); },
                                 state_.get(), "wait");
        }
        complete();
    }

    bool test() override {
        if (status_ == Status::SUCCESS) return true;  // completed before any abort
        if (recorded_) {
            detail::throwIfAborted(state_.get(), "test");
            cudaError_t err = cudaEventQuery(event_);
            if (err == cudaErrorNotReady) {
                // A caller looping on test() must see a failed communicator
                // too, or it would spin forever on an event that never fires.
                if (state_ && state_->comm) {
                    ncclResult_t async_err = ncclSuccess;
                    STCOMM_NCCL_CHECK(ncclCommGetAsyncError(state_->comm, &async_err));
                    if (async_err != ncclSuccess && async_err != ncclInProgress) {
                        detail::abortAndThrow(*state_, std::string("test: communicator failed: ") +
                                                       ncclGetErrorString(async_err));
                    }
                }
                return false;
            }
            STCOMM_CUDA_CHECK(err);
        }
        complete();
        return true;
    }

    Status getStatus() const override { return status_; }
    Backend getBackend() const override { return Backend::NCCL; }

    /// @brief Stream this request was recorded on (null until record()). For
    /// advanced callers chaining their own CUDA work behind the operation.
    cudaStream_t getStream() const { return stream_; }

private:
    void complete() {
        if (status_ == Status::SUCCESS) return;
        status_ = Status::SUCCESS;
        if (finalizer_) finalizer_();
    }

    cudaStream_t          stream_ = nullptr;
    cudaEvent_t           event_  = nullptr;
    std::shared_ptr<detail::NCCLCommState> state_;
    bool                  recorded_ = false;
    std::shared_ptr<void> scratch_;
    std::function<void()> finalizer_;
    Status                status_;
};

using MPIRequestPtr  = std::shared_ptr<MPIRequest>;
using NCCLRequestPtr = std::shared_ptr<NCCLRequest>;

} // namespace stComm
