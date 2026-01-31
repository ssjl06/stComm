#ifndef STCOMM_NCCL_COMM_H
#define STCOMM_NCCL_COMM_H

#include "types.h"
#include "request.h"
#include "utils.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <type_traits>

namespace stComm {

// NCCL datatype mapper for primitive types
template<typename T>
struct NCCLTypeMap {
    static ncclDataType_t type();
};

// Specializations for common types
template<> inline ncclDataType_t NCCLTypeMap<int8_t>::type() { return ncclInt8; }
template<> inline ncclDataType_t NCCLTypeMap<uint8_t>::type() { return ncclUint8; }
template<> inline ncclDataType_t NCCLTypeMap<int32_t>::type() { return ncclInt32; }
template<> inline ncclDataType_t NCCLTypeMap<uint32_t>::type() { return ncclUint32; }
template<> inline ncclDataType_t NCCLTypeMap<int64_t>::type() { return ncclInt64; }
template<> inline ncclDataType_t NCCLTypeMap<uint64_t>::type() { return ncclUint64; }
template<> inline ncclDataType_t NCCLTypeMap<float>::type() { return ncclFloat32; }
template<> inline ncclDataType_t NCCLTypeMap<double>::type() { return ncclFloat64; }
template<> inline ncclDataType_t NCCLTypeMap<char>::type() { return ncclInt8; }
template<> inline ncclDataType_t NCCLTypeMap<int>::type() { return ncclInt32; }

class NCCLComm {
public:
    NCCLComm();
    ~NCCLComm();

    // Initialize NCCL communicator
    // rank: process rank
    // nranks: total number of processes
    // device_id: CUDA device ID for this process
    void initialize(int rank, int nranks, int device_id, ncclUniqueId comm_id);

    // Get unique ID for communicator (call on rank 0 and broadcast)
    static ncclUniqueId getUniqueId();

    // Get rank and size
    int getRank() const { return rank_; }
    int getSize() const { return size_; }

    // Point-to-point communication (async)
    template<typename T>
    RequestPtr send(const T* data, size_t count, int dest, cudaStream_t stream = nullptr);

    template<typename T>
    RequestPtr recv(T* data, size_t count, int source, cudaStream_t stream = nullptr);

    // Collective communication (async)
    template<typename T>
    RequestPtr allgather(const T* sendbuf, T* recvbuf, size_t count, cudaStream_t stream = nullptr);

    template<typename T>
    RequestPtr alltoall(const T* sendbuf, T* recvbuf, size_t count, cudaStream_t stream = nullptr);

    // Variable-length collectives (implemented via grouped operations)
    template<typename T>
    RequestPtr allgatherv(const T* sendbuf, size_t sendcount, T* recvbuf,
                         const int* recvcounts, const int* displs, cudaStream_t stream = nullptr);

    template<typename T>
    RequestPtr allgatherv_auto(const T* sendbuf, size_t sendcount, T* recvbuf,
                               const int* recvcounts, cudaStream_t stream = nullptr);

    template<typename T>
    RequestPtr alltoallv(const T* sendbuf, const int* sendcounts, const int* sdispls,
                        T* recvbuf, const int* recvcounts, const int* rdispls,
                        cudaStream_t stream = nullptr);

    template<typename T>
    RequestPtr alltoallv_auto(const T* sendbuf, const int* sendcounts,
                              T* recvbuf, const int* recvcounts, cudaStream_t stream = nullptr);

    // Reduce operations
    template<typename T>
    RequestPtr allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ncclRedOp_t op = ncclSum, cudaStream_t stream = nullptr);

    ncclComm_t getHandle() const { return comm_; }

private:
    ncclComm_t comm_;
    int rank_;
    int size_;
    int device_id_;
    bool initialized_;

    cudaStream_t getOrCreateStream(cudaStream_t stream);
};

// Template implementations
template<typename T>
RequestPtr NCCLComm::send(const T* data, size_t count, int dest, cudaStream_t stream) {
    if (!initialized_) {
        return nullptr;
    }

    cudaStream_t use_stream = getOrCreateStream(stream);
    auto req = std::make_shared<NCCLRequest>(use_stream);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    ncclSend(data, count, nccl_type, dest, comm_, use_stream);

    return req;
}

template<typename T>
RequestPtr NCCLComm::recv(T* data, size_t count, int source, cudaStream_t stream) {
    if (!initialized_) {
        return nullptr;
    }

    cudaStream_t use_stream = getOrCreateStream(stream);
    auto req = std::make_shared<NCCLRequest>(use_stream);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    ncclRecv(data, count, nccl_type, source, comm_, use_stream);

    return req;
}

template<typename T>
RequestPtr NCCLComm::allgather(const T* sendbuf, T* recvbuf, size_t count, cudaStream_t stream) {
    if (!initialized_) {
        return nullptr;
    }

    cudaStream_t use_stream = getOrCreateStream(stream);
    auto req = std::make_shared<NCCLRequest>(use_stream);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    ncclAllGather(sendbuf, recvbuf, count, nccl_type, comm_, use_stream);

    return req;
}

template<typename T>
RequestPtr NCCLComm::alltoall(const T* sendbuf, T* recvbuf, size_t count, cudaStream_t stream) {
    if (!initialized_) {
        return nullptr;
    }

    // NCCL doesn't have native alltoall, implement using grouped operations
    cudaStream_t use_stream = getOrCreateStream(stream);
    auto req = std::make_shared<NCCLRequest>(use_stream);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();

    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
        ncclSend(sendbuf + i * count, count, nccl_type, i, comm_, use_stream);
        ncclRecv(recvbuf + i * count, count, nccl_type, i, comm_, use_stream);
    }
    ncclGroupEnd();

    return req;
}

template<typename T>
RequestPtr NCCLComm::allreduce(const T* sendbuf, T* recvbuf, size_t count,
                               ncclRedOp_t op, cudaStream_t stream) {
    if (!initialized_) {
        return nullptr;
    }

    cudaStream_t use_stream = getOrCreateStream(stream);
    auto req = std::make_shared<NCCLRequest>(use_stream);

    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();
    ncclAllReduce(sendbuf, recvbuf, count, nccl_type, op, comm_, use_stream);

    return req;
}

template<typename T>
RequestPtr NCCLComm::allgatherv(const T* sendbuf, size_t sendcount, T* recvbuf,
                                const int* recvcounts, const int* displs, cudaStream_t stream) {
    if (!initialized_) {
        return nullptr;
    }

    cudaStream_t use_stream = getOrCreateStream(stream);
    auto req = std::make_shared<NCCLRequest>(use_stream);
    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();

    // NCCL doesn't have native allgatherv, use grouped send/recv
    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
        if (i == rank_) {
            // Broadcast my data to all others
            for (int j = 0; j < size_; ++j) {
                if (j != rank_) {
                    ncclSend(sendbuf, sendcount, nccl_type, j, comm_, use_stream);
                }
            }
            // Copy my own data (can be done with cudaMemcpy or in-place)
            if (recvbuf + displs[rank_] != sendbuf) {
                cudaMemcpyAsync(recvbuf + displs[rank_], sendbuf, sendcount * sizeof(T),
                               cudaMemcpyDeviceToDevice, use_stream);
            }
        } else {
            // Receive from rank i
            ncclRecv(recvbuf + displs[i], recvcounts[i], nccl_type, i, comm_, use_stream);
        }
    }
    ncclGroupEnd();

    return req;
}

template<typename T>
RequestPtr NCCLComm::allgatherv_auto(const T* sendbuf, size_t sendcount, T* recvbuf,
                                      const int* recvcounts, cudaStream_t stream) {
    auto displs = Utils::calculate_displacements(recvcounts, size_);
    return allgatherv(sendbuf, sendcount, recvbuf, recvcounts, displs.data(), stream);
}

template<typename T>
RequestPtr NCCLComm::alltoallv(const T* sendbuf, const int* sendcounts, const int* sdispls,
                               T* recvbuf, const int* recvcounts, const int* rdispls,
                               cudaStream_t stream) {
    if (!initialized_) {
        return nullptr;
    }

    cudaStream_t use_stream = getOrCreateStream(stream);
    auto req = std::make_shared<NCCLRequest>(use_stream);
    ncclDataType_t nccl_type = NCCLTypeMap<T>::type();

    // NCCL doesn't have native alltoallv, use grouped send/recv
    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
        if (i != rank_) {
            if (sendcounts[i] > 0) {
                ncclSend(sendbuf + sdispls[i], sendcounts[i], nccl_type, i, comm_, use_stream);
            }
            if (recvcounts[i] > 0) {
                ncclRecv(recvbuf + rdispls[i], recvcounts[i], nccl_type, i, comm_, use_stream);
            }
        } else {
            // Self-copy for same rank
            if (sendcounts[rank_] > 0 && (recvbuf + rdispls[rank_] != sendbuf + sdispls[rank_])) {
                cudaMemcpyAsync(recvbuf + rdispls[rank_], sendbuf + sdispls[rank_],
                               sendcounts[rank_] * sizeof(T),
                               cudaMemcpyDeviceToDevice, use_stream);
            }
        }
    }
    ncclGroupEnd();

    return req;
}

template<typename T>
RequestPtr NCCLComm::alltoallv_auto(const T* sendbuf, const int* sendcounts,
                                     T* recvbuf, const int* recvcounts, cudaStream_t stream) {
    auto sdispls = Utils::calculate_displacements(sendcounts, size_);
    auto rdispls = Utils::calculate_displacements(recvcounts, size_);
    return alltoallv(sendbuf, sendcounts, sdispls.data(),
                     recvbuf, recvcounts, rdispls.data(), stream);
}

} // namespace stComm

#endif // STCOMM_NCCL_COMM_H
