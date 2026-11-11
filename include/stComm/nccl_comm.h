#pragma once

#include "types.h"
#include "request.h"
#include "utils.h"
#include "cuda_check.h"
#include "nccl_check.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stComm {

namespace detail {

// Apply a ReduceOp to two host-side values. Used by NCCLComm's emulated
// reductions after the gathered values have been copied back to the host.
template<typename T>
inline T apply_reduce(ReduceOp op, T a, T b) {
    switch (op) {
        case ReduceOp::Sum:  return a + b;
        case ReduceOp::Prod: return a * b;
        case ReduceOp::Max:  return b > a ? b : a;
        case ReduceOp::Min:  return b < a ? b : a;
    }
    return a;  // unreachable
}

// Map the backend-agnostic ReduceOp to the native NCCL reduction op. Kept
// internal so callers of NCCLComm never name ncclRedOp_t themselves.
inline ncclRedOp_t to_nccl_op(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum:  return ncclSum;
        case ReduceOp::Prod: return ncclProd;
        case ReduceOp::Max:  return ncclMax;
        case ReduceOp::Min:  return ncclMin;
    }
    return ncclSum;  // unreachable; silences -Wreturn-type
}

}  // namespace detail

/**
 * @brief NCCL datatype mapper for primitive types
 *
 * Unmapped types are a compile error (deleted primary), like
 * detail::mpi_datatype. NCCL has no 16-bit datatype, so int16_t/uint16_t are
 * deliberately not mapped: sending them as a 32-bit type would move 4 bytes per
 * 2-byte element and overrun the buffer.
 */
template<typename T>
struct NCCLTypeMap {
    static ncclDataType_t type() = delete;
};

// Specializations for fixed-size types
template<> inline ncclDataType_t NCCLTypeMap<int8_t>::type() { return ncclInt8; }
template<> inline ncclDataType_t NCCLTypeMap<uint8_t>::type() { return ncclUint8; }
template<> inline ncclDataType_t NCCLTypeMap<int32_t>::type() { return ncclInt32; }
template<> inline ncclDataType_t NCCLTypeMap<uint32_t>::type() { return ncclUint32; }
template<> inline ncclDataType_t NCCLTypeMap<int64_t>::type() { return ncclInt64; }
template<> inline ncclDataType_t NCCLTypeMap<uint64_t>::type() { return ncclUint64; }
template<> inline ncclDataType_t NCCLTypeMap<float>::type() { return ncclFloat32; }
template<> inline ncclDataType_t NCCLTypeMap<double>::type() { return ncclFloat64; }

namespace detail {
template<typename T> struct GatherScratch;
}  // namespace detail

/**
 * @brief NCCL-based communication backend for GPU
 *
 * Implements communication using NCCL for device memory.
 * All data pointers must point to GPU memory.
 */
class NCCLComm {
public:
    NCCLComm();
    ~NCCLComm();

    // Owns a ncclComm_t + CUDA stream that the destructor frees, so copying
    // would double-free — non-copyable.
    NCCLComm(const NCCLComm&)            = delete;
    NCCLComm& operator=(const NCCLComm&) = delete;

    // Initialization
    void initialize(int rank, int nranks, int device_id, ncclUniqueId comm_id);
    static ncclUniqueId getUniqueId();

    // Scalar accessors
    int getRank() const { return rank_; }
    int getSize() const { return size_; }
    Backend getBackend() const { return Backend::NCCL; }
    void barrier();

    // Point-to-point communication (async, device memory).
    // Enqueued on the internal CUDA stream; the returned request records its own
    // event for per-request completion (see request.h). For a matched send+recv
    // pair, wrap them in groupStart()/groupEnd() to avoid deadlock — the events
    // are then recorded at groupEnd(), once the ops are flushed to the stream.
    template<typename T>
    std::shared_ptr<NCCLRequest> send(const T* data, size_t count, int dest);

    template<typename T>
    std::shared_ptr<NCCLRequest> recv(T* data, size_t count, int source);

    // Group management for batching multiple operations. Between groupStart()
    // and groupEnd() the individual ops are fused and not flushed to the stream
    // until groupEnd(); requests issued in that window defer their event record
    // to groupEnd(), so wait()/test() on them is only meaningful afterwards.
    void groupStart();
    void groupEnd();

    // Collective communication - with auto displacement (device memory).
    // Enqueued on the internal CUDA stream; the request records its own event.
    template<typename T>
    std::shared_ptr<NCCLRequest> allgatherv(const T* sendbuf, int sendcount,
                         T* recvbuf, const int* recvcounts);

    template<typename T>
    std::shared_ptr<NCCLRequest> alltoallv(const T* sendbuf, const int* sendcounts,
                        T* recvbuf, const int* recvcounts);

    // size_t-count variants: no INT_MAX limit on a count or on a side's total
    // (NCCL takes size_t counts; offsets are pointer arithmetic). The int
    // versions above forward here.
    template<typename T>
    std::shared_ptr<NCCLRequest> allgatherv(const T* sendbuf, size_t sendcount,
                         T* recvbuf, const size_t* recvcounts);

    template<typename T>
    std::shared_ptr<NCCLRequest> alltoallv(const T* sendbuf, const size_t* sendcounts,
                        T* recvbuf, const size_t* recvcounts);

    // Broadcast `count` elements of T from `root` to all ranks (async,
    // device memory). In-place: `data` is both source on root and destination
    // on every rank. Issued on the internal CUDA stream; call .wait() to
    // synchronize.
    template<typename T>
    std::shared_ptr<NCCLRequest> bcast(T* data, size_t count, int root);

    // Emulated reductions — NCCL has no MAXLOC or scan primitive, so these
    // ncclAllGather every rank's scalar and reduce on the host (type-agnostic,
    // exact). Async: the scalar result lands in *out once the returned request
    // completes; the gather buffers live inside the request until then. Same
    // shape as MPIComm's async reduction overloads, so the Comm facade routes
    // both by Space tag. Not usable inside groupStart()/groupEnd() (the gather
    // would be deferred past the copy that reads its result) — throws
    // std::logic_error there.
    template<typename T>
    std::shared_ptr<NCCLRequest> allreduceMaxloc(T value, std::pair<T, int>* out);

    template<typename T>
    std::shared_ptr<NCCLRequest> exscan(T value, T* out, ReduceOp op = ReduceOp::Sum);

    // Element-wise allreduce over `count` values of T (async, device memory):
    // recvbuf receives the reduction (op) of every rank's sendbuf. Native
    // ncclAllReduce — no host staging. In-place is allowed (sendbuf == recvbuf).
    // Issued on the internal CUDA stream; call .wait() to synchronize.
    template<typename T>
    std::shared_ptr<NCCLRequest> allreduce(const T* sendbuf, T* recvbuf,
                                           size_t count, ReduceOp op = ReduceOp::Sum);

    // Get native handle
    ncclComm_t getHandle() const { return comm_; }

    // The internal CUDA stream every NCCL operation is enqueued on. Advanced
    // callers launch their own kernels on this stream to interleave compute
    // with collectives in strict stream order — no event juggling, no
    // per-operation synchronization (the device-resident solver loop pattern).
    cudaStream_t getStream() const { return stream_; }

private:
    // Every submission starts here: using the communicator before
    // initialize() is a caller bug, and an aborted one accepts no more work.
    void requireSubmittable() const {
        if (!initialized_) {
            throw std::logic_error("stComm NCCL: communicator used before initialize()");
        }
        detail::throwIfAborted(state_.get(), "submit");
    }

    // Shared front half of the emulated reductions: gather every rank's
    // `value` into the returned scratch's host buffer, on the stream.
    template<typename T>
    std::shared_ptr<detail::GatherScratch<T>> enqueueGather(T value, const char* what);

    // Record the request's completion event now, or — if we are inside a user
    // groupStart()/groupEnd() — defer it to groupEnd(), since the ops are not
    // actually on the stream until the group is flushed.
    void recordOrDefer(const std::shared_ptr<NCCLRequest>& req);

    ncclComm_t comm_;
    // Shared with every issued request (see detail::NCCLCommState): lets a
    // wait that detects an async failure abort the communicator, and tells
    // the destructor not to destroy an aborted one.
    std::shared_ptr<detail::NCCLCommState> state_;
    int rank_;
    int size_;
    int device_id_;
    bool initialized_;
    cudaStream_t stream_;  // Internal CUDA stream for all operations

    // User-group bookkeeping (set by groupStart/groupEnd). Requests created
    // while in_group_ is true park here until groupEnd() records their events.
    bool in_group_ = false;
    std::vector<std::shared_ptr<NCCLRequest>> pending_records_;
};

// ============================================================================
// Template implementations
// ============================================================================

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::send(const T* data, size_t count, int dest) {
    requireSubmittable();

    auto req = std::make_shared<NCCLRequest>();

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    STCOMM_NCCL_CHECK(ncclSend(data, count, nccl_type, dest, comm_, stream_));

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::recv(T* data, size_t count, int source) {
    requireSubmittable();

    auto req = std::make_shared<NCCLRequest>();

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    STCOMM_NCCL_CHECK(ncclRecv(data, count, nccl_type, source, comm_, stream_));

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::allgatherv(const T* sendbuf, int sendcount,
                                T* recvbuf, const int* recvcounts) {
    if (sendcount < 0) {
        throw std::overflow_error("stComm NCCL allgatherv: negative sendcount");
    }
    const auto counts = Utils::toSizeCounts(recvcounts, size_);
    return allgatherv(sendbuf, static_cast<size_t>(sendcount), recvbuf, counts.data());
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::alltoallv(const T* sendbuf, const int* sendcounts,
                               T* recvbuf, const int* recvcounts) {
    const auto scounts = Utils::toSizeCounts(sendcounts, size_);
    const auto rcounts = Utils::toSizeCounts(recvcounts, size_);
    return alltoallv(sendbuf, scounts.data(), recvbuf, rcounts.data());
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::allgatherv(const T* sendbuf, size_t sendcount,
                                T* recvbuf, const size_t* recvcounts) {
    requireSubmittable();

    auto req = std::make_shared<NCCLRequest>();
    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    const auto nranks = static_cast<size_t>(size_);
    const auto me     = static_cast<size_t>(rank_);

    // NCCL doesn't have native allgatherv, use grouped send/recv. Offsets are
    // plain pointer arithmetic, so there is no int displacement limit.
    STCOMM_NCCL_CHECK(ncclGroupStart());
    size_t offset = 0;  // in elements
    for (size_t i = 0; i < nranks; ++i) {
        T* dst = recvbuf + offset;
        if (i == me) {
            // Broadcast my data to all others
            for (size_t j = 0; j < nranks; ++j) {
                if (j != me) {
                    STCOMM_NCCL_CHECK(ncclSend(sendbuf, sendcount, nccl_type,
                                               static_cast<int>(j), comm_, stream_));
                }
            }
            // Copy my own data
            if (dst != sendbuf) {
                STCOMM_CUDA_CHECK(cudaMemcpyAsync(dst, sendbuf, sendcount * sizeof(T),
                                                  cudaMemcpyDeviceToDevice, stream_));
            }
        } else {
            // Receive from rank i
            STCOMM_NCCL_CHECK(ncclRecv(dst, recvcounts[i], nccl_type,
                                       static_cast<int>(i), comm_, stream_));
        }
        offset += recvcounts[i];
    }
    STCOMM_NCCL_CHECK(ncclGroupEnd());

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::alltoallv(const T* sendbuf, const size_t* sendcounts,
                               T* recvbuf, const size_t* recvcounts) {
    requireSubmittable();

    auto req = std::make_shared<NCCLRequest>();
    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    const auto nranks = static_cast<size_t>(size_);
    const auto me     = static_cast<size_t>(rank_);

    // NCCL doesn't have native alltoallv, use grouped send/recv. Offsets are
    // plain pointer arithmetic, so there is no int displacement limit.
    STCOMM_NCCL_CHECK(ncclGroupStart());
    size_t soff = 0, roff = 0;  // in elements
    for (size_t i = 0; i < nranks; ++i) {
        const T* src = sendbuf + soff;
        T* dst = recvbuf + roff;
        if (i != me) {
            if (sendcounts[i] > 0) {
                STCOMM_NCCL_CHECK(ncclSend(src, sendcounts[i], nccl_type,
                                           static_cast<int>(i), comm_, stream_));
            }
            if (recvcounts[i] > 0) {
                STCOMM_NCCL_CHECK(ncclRecv(dst, recvcounts[i], nccl_type,
                                           static_cast<int>(i), comm_, stream_));
            }
        } else {
            // Self-copy
            if (sendcounts[i] > 0 && dst != src) {
                STCOMM_CUDA_CHECK(cudaMemcpyAsync(dst, src, sendcounts[i] * sizeof(T),
                                                  cudaMemcpyDeviceToDevice, stream_));
            }
        }
        soff += sendcounts[i];
        roff += recvcounts[i];
    }
    STCOMM_NCCL_CHECK(ncclGroupEnd());

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::bcast(T* data, size_t count, int root) {
    requireSubmittable();

    auto req = std::make_shared<NCCLRequest>();

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    STCOMM_NCCL_CHECK(ncclBroadcast(data, data, count, nccl_type, root, comm_, stream_));

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::allreduce(const T* sendbuf, T* recvbuf,
                                                 size_t count, ReduceOp op) {
    requireSubmittable();

    auto req = std::make_shared<NCCLRequest>();

    STCOMM_NCCL_CHECK(ncclAllReduce(sendbuf, recvbuf, count, NCCLTypeMap<T>::type(),
                                    detail::to_nccl_op(op), comm_, stream_));

    recordOrDefer(req);
    return req;
}

namespace detail {

// Scratch for an emulated reduction, owned by the request via setScratch() and
// freed when the request (and thus this scratch) is released:
//   dev  — device buffer of 1 + nranks elements: [0] is this rank's value (the
//          gather's send side), [1..] receives everyone's values.
//   host — the same layout on the host, *page-locked*. The copies to and from
//          it are enqueued behind NCCL work on the stream; with pageable
//          memory cudaMemcpyAsync would block inside the submitting call until
//          that work finished, bypassing the request's polled wait (and its
//          failure detection) altogether.
template<typename T>
struct GatherScratch {
    T* dev  = nullptr;
    T* host = nullptr;
    ~GatherScratch() {
        cudaFree(dev);
        cudaFreeHost(host);
    }
    const T* gathered() const { return host + 1; }
};

}  // namespace detail

template<typename T>
std::shared_ptr<detail::GatherScratch<T>> NCCLComm::enqueueGather(T value, const char* what) {
    requireSubmittable();
    if (in_group_) {
        throw std::logic_error(std::string("stComm NCCL: ") + what +
                               " cannot be issued inside groupStart()/groupEnd()");
    }

    auto s = std::make_shared<detail::GatherScratch<T>>();
    const auto nranks = static_cast<size_t>(size_);
    const size_t bytes = (1 + nranks) * sizeof(T);
    STCOMM_CUDA_CHECK(cudaMalloc(&s->dev, bytes));
    STCOMM_CUDA_CHECK(cudaMallocHost(&s->host, bytes));
    s->host[0] = value;

    STCOMM_CUDA_CHECK(cudaMemcpyAsync(s->dev, s->host, sizeof(T),
                                      cudaMemcpyHostToDevice, stream_));
    STCOMM_NCCL_CHECK(ncclAllGather(s->dev, s->dev + 1, 1, NCCLTypeMap<T>::type(),
                                    comm_, stream_));
    STCOMM_CUDA_CHECK(cudaMemcpyAsync(s->host + 1, s->dev + 1, nranks * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream_));
    return s;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::allreduceMaxloc(T value, std::pair<T, int>* out) {
    auto s = enqueueGather(value, "allreduceMaxloc");
    const auto nranks = static_cast<size_t>(size_);

    auto req = std::make_shared<NCCLRequest>();
    req->setScratch(s);
    // Runs once after the stream drains: argmax over the gathered values. Ties
    // go to the smaller rank (strict >), matching MPI_MAXLOC semantics.
    req->setFinalizer([s, out, nranks]() {
        const T* v = s->gathered();
        size_t best = 0;
        for (size_t i = 1; i < nranks; ++i)
            if (v[i] > v[best]) best = i;
        *out = { v[best], static_cast<int>(best) };
    });
    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::exscan(T value, T* out, ReduceOp op) {
    // Same gather as allreduceMaxloc; the difference is the host-side reduce.
    auto s = enqueueGather(value, "exscan");

    auto req = std::make_shared<NCCLRequest>();
    req->setScratch(s);
    const auto rank = static_cast<size_t>(rank_);
    // Runs once after the stream drains: exclusive scan over ranks [0, rank).
    // Rank 0 returns T{} to match MPIComm::exscan (MPI_Exscan leaves rank 0
    // undefined).
    req->setFinalizer([s, out, op, rank]() {
        if (rank == 0) { *out = T{}; return; }
        const T* v = s->gathered();
        T acc = v[0];
        for (size_t i = 1; i < rank; ++i)
            acc = detail::apply_reduce(op, acc, v[i]);
        *out = acc;
    });
    recordOrDefer(req);
    return req;
}

} // namespace stComm
