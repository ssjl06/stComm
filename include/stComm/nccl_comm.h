#pragma once

#include "comm_base.h"
#include "utils.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>

namespace stComm {

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
class NCCLComm : public CommBase {
public:
    NCCLComm();
    ~NCCLComm() override;

    // Initialization
    void initialize(int rank, int nranks, int device_id, ncclUniqueId comm_id);
    static ncclUniqueId getUniqueId();

    // CommBase interface
    int getRank() const override { return rank_; }
    int getSize() const override { return size_; }
    Backend getBackend() const override { return Backend::NCCL; }
    void barrier() override;

    // Point-to-point communication (async, device memory)
    // Uses internal CUDA stream created during initialization
    template<typename T>
    RequestPtr send(const T* data, size_t count, int dest);

    template<typename T>
    RequestPtr recv(T* data, size_t count, int source);

    // Collective communication - with auto displacement (device memory)
    // Uses internal CUDA stream created during initialization
    template<typename T>
    RequestPtr allgatherv(const T* sendbuf, int sendcount,
                         T* recvbuf, const int* recvcounts);

    template<typename T>
    RequestPtr alltoallv(const T* sendbuf, const int* sendcounts,
                        T* recvbuf, const int* recvcounts);

    // Get native handle
    ncclComm_t getHandle() const { return comm_; }

private:
    ncclComm_t comm_;
    int rank_;
    int size_;
    int device_id_;
    bool initialized_;
    cudaStream_t stream_;  // Internal CUDA stream for all operations
};

// ============================================================================
// Template implementations
// ============================================================================

template<typename T>
RequestPtr NCCLComm::send(const T* data, size_t count, int dest) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>(stream_);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    ncclSend(data, count, nccl_type, dest, comm_, stream_);

    return req;
}

template<typename T>
RequestPtr NCCLComm::recv(T* data, size_t count, int source) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>(stream_);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    ncclRecv(data, count, nccl_type, source, comm_, stream_);

    return req;
}

template<typename T>
RequestPtr NCCLComm::allgatherv(const T* sendbuf, int sendcount,
                                T* recvbuf, const int* recvcounts) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>(stream_);

    // Auto-calculate displacements
    auto displs = Utils::calculateDisplacements(recvcounts, size_);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();

    // NCCL doesn't have native allgatherv, use grouped send/recv
    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
        if (i == rank_) {
            // Broadcast my data to all others
            for (int j = 0; j < size_; ++j) {
                if (j != rank_) {
                    ncclSend(sendbuf, sendcount, nccl_type, j, comm_, stream_);
                }
            }
            // Copy my own data
            if (recvbuf + displs[rank_] != sendbuf) {
                cudaMemcpyAsync(recvbuf + displs[rank_], sendbuf, sendcount * sizeof(T),
                               cudaMemcpyDeviceToDevice, stream_);
            }
        } else {
            // Receive from rank i
            ncclRecv(recvbuf + displs[i], recvcounts[i], nccl_type, i, comm_, stream_);
        }
    }
    ncclGroupEnd();

    return req;
}

template<typename T>
RequestPtr NCCLComm::alltoallv(const T* sendbuf, const int* sendcounts,
                               T* recvbuf, const int* recvcounts) {
    if (!initialized_) {
        return nullptr;
    }

    auto req = std::make_shared<NCCLRequest>(stream_);

    // Auto-calculate displacements
    auto sdispls = Utils::calculateDisplacements(sendcounts, size_);
    auto rdispls = Utils::calculateDisplacements(recvcounts, size_);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();

    // NCCL doesn't have native alltoallv, use grouped send/recv
    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
        if (i != rank_) {
            if (sendcounts[i] > 0) {
                ncclSend(sendbuf + sdispls[i], sendcounts[i], nccl_type, i, comm_, stream_);
            }
            if (recvcounts[i] > 0) {
                ncclRecv(recvbuf + rdispls[i], recvcounts[i], nccl_type, i, comm_, stream_);
            }
        } else {
            // Self-copy
            if (sendcounts[rank_] > 0 && (recvbuf + rdispls[rank_] != sendbuf + sdispls[rank_])) {
                cudaMemcpyAsync(recvbuf + rdispls[rank_], sendbuf + sdispls[rank_],
                               sendcounts[rank_] * sizeof(T),
                               cudaMemcpyDeviceToDevice, stream_);
            }
        }
    }
    ncclGroupEnd();

    return req;
}

} // namespace stComm
