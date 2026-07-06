#pragma once

#include "types.h"
#include "request.h"
#include "utils.h"
#include "cuda_check.h"
#include "nccl_check.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
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
 */
template<typename T>
struct NCCLTypeMap {
    static ncclDataType_t type();
};

// Specializations for fixed-size types
template<> inline ncclDataType_t NCCLTypeMap<int8_t>::type() { return ncclInt8; }
template<> inline ncclDataType_t NCCLTypeMap<uint8_t>::type() { return ncclUint8; }
template<> inline ncclDataType_t NCCLTypeMap<int16_t>::type() { return ncclInt32; }  // NCCL doesn't have int16, use int32
template<> inline ncclDataType_t NCCLTypeMap<uint16_t>::type() { return ncclUint32; }  // NCCL doesn't have uint16, use uint32
template<> inline ncclDataType_t NCCLTypeMap<int32_t>::type() { return ncclInt32; }
template<> inline ncclDataType_t NCCLTypeMap<uint32_t>::type() { return ncclUint32; }
template<> inline ncclDataType_t NCCLTypeMap<int64_t>::type() { return ncclInt64; }
template<> inline ncclDataType_t NCCLTypeMap<uint64_t>::type() { return ncclUint64; }
template<> inline ncclDataType_t NCCLTypeMap<float>::type() { return ncclFloat32; }
template<> inline ncclDataType_t NCCLTypeMap<double>::type() { return ncclFloat64; }

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
    // both by Space tag.
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
    // Record the request's completion event now, or — if we are inside a user
    // groupStart()/groupEnd() — defer it to groupEnd(), since the ops are not
    // actually on the stream until the group is flushed.
    void recordOrDefer(const std::shared_ptr<NCCLRequest>& req);

    ncclComm_t comm_;
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
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>();

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    STCOMM_NCCL_CHECK(ncclSend(data, count, nccl_type, dest, comm_, stream_));

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::recv(T* data, size_t count, int source) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>();

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    STCOMM_NCCL_CHECK(ncclRecv(data, count, nccl_type, source, comm_, stream_));

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::allgatherv(const T* sendbuf, int sendcount,
                                T* recvbuf, const int* recvcounts) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>();

    // Auto-calculate displacements
    auto displs = Utils::calculateDisplacements(recvcounts, size_);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();

    // NCCL doesn't have native allgatherv, use grouped send/recv
    STCOMM_NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < size_; ++i) {
        if (i == rank_) {
            // Broadcast my data to all others
            for (int j = 0; j < size_; ++j) {
                if (j != rank_) {
                    STCOMM_NCCL_CHECK(ncclSend(sendbuf, sendcount, nccl_type, j, comm_, stream_));
                }
            }
            // Copy my own data
            if (recvbuf + displs[rank_] != sendbuf) {
                STCOMM_CUDA_CHECK(cudaMemcpyAsync(recvbuf + displs[rank_], sendbuf,
                                                  sendcount * sizeof(T),
                                                  cudaMemcpyDeviceToDevice, stream_));
            }
        } else {
            // Receive from rank i
            STCOMM_NCCL_CHECK(ncclRecv(recvbuf + displs[i], recvcounts[i], nccl_type, i, comm_, stream_));
        }
    }
    STCOMM_NCCL_CHECK(ncclGroupEnd());

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::alltoallv(const T* sendbuf, const int* sendcounts,
                               T* recvbuf, const int* recvcounts) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>();

    // Auto-calculate displacements
    auto sdispls = Utils::calculateDisplacements(sendcounts, size_);
    auto rdispls = Utils::calculateDisplacements(recvcounts, size_);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();

    // NCCL doesn't have native alltoallv, use grouped send/recv
    STCOMM_NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < size_; ++i) {
        if (i != rank_) {
            if (sendcounts[i] > 0) {
                STCOMM_NCCL_CHECK(ncclSend(sendbuf + sdispls[i], sendcounts[i], nccl_type, i, comm_, stream_));
            }
            if (recvcounts[i] > 0) {
                STCOMM_NCCL_CHECK(ncclRecv(recvbuf + rdispls[i], recvcounts[i], nccl_type, i, comm_, stream_));
            }
        } else {
            // Self-copy
            if (sendcounts[rank_] > 0 && (recvbuf + rdispls[rank_] != sendbuf + sdispls[rank_])) {
                STCOMM_CUDA_CHECK(cudaMemcpyAsync(recvbuf + rdispls[rank_], sendbuf + sdispls[rank_],
                                                  sendcounts[rank_] * sizeof(T),
                                                  cudaMemcpyDeviceToDevice, stream_));
            }
        }
    }
    STCOMM_NCCL_CHECK(ncclGroupEnd());

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::bcast(T* data, size_t count, int root) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>();

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    STCOMM_NCCL_CHECK(ncclBroadcast(data, data, count, nccl_type, root, comm_, stream_));

    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::allreduce(const T* sendbuf, T* recvbuf,
                                                 size_t count, ReduceOp op) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>();

    STCOMM_NCCL_CHECK(ncclAllReduce(sendbuf, recvbuf, count, NCCLTypeMap<T>::type(),
                                    detail::to_nccl_op(op), comm_, stream_));

    recordOrDefer(req);
    return req;
}

namespace detail {

// Scratch for an emulated reduction, owned by the request via setScratch():
//   hsend   — a stable host copy of the caller's by-value input (the async
//             H2D copy reads from here, so it must outlive the call, not the
//             stack parameter).
//   d_send/d_recv — device send/gather buffers, freed in the destructor when
//             the request (and thus this scratch) is released.
//   host    — gathered values copied back for the host-side reduce.
template<typename T>
struct GatherScratch {
    T  hsend{};
    T* d_send = nullptr;
    T* d_recv = nullptr;
    std::vector<T> host;
    ~GatherScratch() {
        cudaFree(d_send);
        cudaFree(d_recv);
    }
};

}  // namespace detail

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::allreduceMaxloc(T value, std::pair<T, int>* out) {
    if (!initialized_) {
        return nullptr;
    }

    // Stage the scalar on the host, gather everyone's value over NCCL, and copy
    // the result back — all enqueued on the internal stream. The scratch keeps
    // these buffers alive until the request completes (see request.h).
    auto s = std::make_shared<detail::GatherScratch<T>>();
    s->hsend = value;
    s->host.resize(size_);
    STCOMM_CUDA_CHECK(cudaMalloc(&s->d_send, sizeof(T)));
    STCOMM_CUDA_CHECK(cudaMalloc(&s->d_recv, size_ * sizeof(T)));

    STCOMM_CUDA_CHECK(cudaMemcpyAsync(s->d_send, &s->hsend, sizeof(T), cudaMemcpyHostToDevice, stream_));
    STCOMM_NCCL_CHECK(ncclAllGather(s->d_send, s->d_recv, 1, NCCLTypeMap<T>::type(), comm_, stream_));
    STCOMM_CUDA_CHECK(cudaMemcpyAsync(s->host.data(), s->d_recv, size_ * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream_));

    auto req = std::make_shared<NCCLRequest>();
    req->setScratch(s);
    const int n = size_;
    // Runs once after the stream drains: argmax over the gathered values. Ties
    // go to the smaller rank (strict >), matching MPI_MAXLOC semantics.
    req->setFinalizer([s, out, n]() {
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (s->host[i] > s->host[best]) best = i;
        *out = { s->host[best], best };
    });
    recordOrDefer(req);
    return req;
}

template<typename T>
std::shared_ptr<NCCLRequest> NCCLComm::exscan(T value, T* out, ReduceOp op) {
    if (!initialized_) {
        return nullptr;
    }

    // Same gather as allreduceMaxloc; the difference is the host-side reduce.
    auto s = std::make_shared<detail::GatherScratch<T>>();
    s->hsend = value;
    s->host.resize(size_);
    STCOMM_CUDA_CHECK(cudaMalloc(&s->d_send, sizeof(T)));
    STCOMM_CUDA_CHECK(cudaMalloc(&s->d_recv, size_ * sizeof(T)));

    STCOMM_CUDA_CHECK(cudaMemcpyAsync(s->d_send, &s->hsend, sizeof(T), cudaMemcpyHostToDevice, stream_));
    STCOMM_NCCL_CHECK(ncclAllGather(s->d_send, s->d_recv, 1, NCCLTypeMap<T>::type(), comm_, stream_));
    STCOMM_CUDA_CHECK(cudaMemcpyAsync(s->host.data(), s->d_recv, size_ * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream_));

    auto req = std::make_shared<NCCLRequest>();
    req->setScratch(s);
    const int rank = rank_;
    // Runs once after the stream drains: exclusive scan over ranks [0, rank).
    // Rank 0 returns T{} to match MPIComm::exscan (MPI_Exscan leaves rank 0
    // undefined).
    req->setFinalizer([s, out, op, rank]() {
        if (rank == 0) { *out = T{}; return; }
        T acc = s->host[0];
        for (int i = 1; i < rank; ++i)
            acc = detail::apply_reduce(op, acc, s->host[i]);
        *out = acc;
    });
    recordOrDefer(req);
    return req;
}

} // namespace stComm
