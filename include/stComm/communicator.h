#ifndef STCOMM_COMMUNICATOR_H
#define STCOMM_COMMUNICATOR_H

#include "comm_base.h"
#include "mpi_comm.h"
#include "nccl_comm.h"
#include "request.h"
#include "utils.h"
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>

namespace stComm {

/**
 * @brief Unified communicator interface supporting both MPI and NCCL backends
 *
 * This class provides a unified interface for both CPU (MPI) and GPU (NCCL)
 * communication. The backend is selected at initialization time.
 *
 * Important memory requirements:
 * - MPI backend: All data pointers must point to HOST memory
 * - NCCL backend: All data pointers must point to DEVICE (GPU) memory
 *
 * Usage example:
 * @code
 *   // MPI backend
 *   Communicator comm(Backend::MPI);
 *   std::vector<int> host_data(100);
 *   auto req = comm.send(host_data.data(), 100, dest_rank);
 *
 *   // NCCL backend
 *   Communicator nccl_comm(Backend::NCCL, rank, size, device_id, nccl_id);
 *   int* d_data;
 *   cudaMalloc(&d_data, 100 * sizeof(int));
 *   auto req = nccl_comm.send(d_data, 100, dest_rank);
 * @endcode
 */
class Communicator {
public:
    /**
     * @brief Construct MPI communicator
     *
     * Creates a communicator using MPI backend. MPI must be initialized
     * before calling this constructor (use MPIComm::initialize()).
     *
     * @param backend Must be Backend::MPI
     */
    explicit Communicator(Backend backend = Backend::MPI);

    /**
     * @brief Construct NCCL communicator
     *
     * Creates a communicator using NCCL backend for GPU communication.
     *
     * @param backend Must be Backend::NCCL
     * @param rank Process rank
     * @param nranks Total number of processes
     * @param device_id CUDA device ID for this process
     * @param comm_id NCCL unique ID (obtain from NCCLComm::getUniqueId())
     */
    Communicator(Backend backend, int rank, int nranks, int device_id, ncclUniqueId comm_id);

    ~Communicator() = default;

    // Move semantics
    Communicator(Communicator&&) = default;
    Communicator& operator=(Communicator&&) = default;

    // Disable copy
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    // ========================================
    // Common interface (non-template methods)
    // ========================================

    /**
     * @brief Get process rank
     * @return Rank ID (0 to size-1)
     */
    int getRank() const;

    /**
     * @brief Get total number of processes
     * @return Number of processes
     */
    int getSize() const;

    /**
     * @brief Synchronization barrier
     */
    void barrier();

    /**
     * @brief Get backend type
     * @return Backend::MPI or Backend::NCCL
     */
    Backend getBackend() const { return backend_; }

    /**
     * @brief Check if communicator is initialized
     */
    bool isInitialized() const;

    // ========================================
    // Point-to-point communication
    // ========================================

    /**
     * @brief Asynchronous send
     *
     * @param data Pointer to data (host memory for MPI, device memory for NCCL)
     * @param count Number of elements to send
     * @param dest Destination rank
     * @param tag Message tag (MPI only, ignored for NCCL)
     * @param stream CUDA stream (NCCL only, ignored for MPI)
     * @return Request handle for async operation
     */
    template<typename T>
    RequestPtr send(const T* data, size_t count, int dest, int tag = 0, cudaStream_t stream = nullptr);

    /**
     * @brief Asynchronous receive
     *
     * @param data Pointer to data buffer (host memory for MPI, device memory for NCCL)
     * @param count Number of elements to receive
     * @param source Source rank
     * @param tag Message tag (MPI only, ignored for NCCL)
     * @param stream CUDA stream (NCCL only, ignored for MPI)
     * @return Request handle for async operation
     */
    template<typename T>
    RequestPtr recv(T* data, size_t count, int source, int tag = 0, cudaStream_t stream = nullptr);

    // ========================================
    // Collective communication - Low-level
    // ========================================

    /**
     * @brief Allgather with variable counts (low-level)
     *
     * @param sendbuf Send buffer
     * @param sendcount Number of elements to send
     * @param recvbuf Receive buffer
     * @param recvcounts Array of receive counts (one per rank)
     * @param displs Array of displacements
     * @param stream CUDA stream (NCCL only)
     */
    template<typename T>
    RequestPtr allgatherv(const T* sendbuf, size_t sendcount,
                         T* recvbuf, const int* recvcounts, const int* displs,
                         cudaStream_t stream = nullptr);

    /**
     * @brief Alltoall with variable counts (low-level)
     */
    template<typename T>
    RequestPtr alltoallv(const T* sendbuf, const int* sendcounts, const int* sdispls,
                        T* recvbuf, const int* recvcounts, const int* rdispls,
                        cudaStream_t stream = nullptr);

    // ========================================
    // Collective communication - High-level (auto displacement)
    // ========================================

    /**
     * @brief Allgather with variable counts (auto displacement)
     *
     * Automatically calculates displacements from recvcounts.
     */
    template<typename T>
    RequestPtr allgatherv_auto(const T* sendbuf, size_t sendcount,
                               T* recvbuf, const int* recvcounts,
                               cudaStream_t stream = nullptr);

    /**
     * @brief Alltoall with variable counts (auto displacement)
     *
     * Automatically calculates displacements from send/recv counts.
     */
    template<typename T>
    RequestPtr alltoallv_auto(const T* sendbuf, const int* sendcounts,
                              T* recvbuf, const int* recvcounts,
                              cudaStream_t stream = nullptr);

    // ========================================
    // Collective communication - Vector API (MPI only)
    // ========================================

    /**
     * @brief Allgather with variable counts (vector API)
     *
     * Note: Only available for MPI backend (host memory).
     * Returns result as std::vector.
     */
    template<typename T>
    std::vector<T> allgatherv_vec(const std::vector<T>& sendbuf,
                                   const std::vector<int>& recvcounts);

    /**
     * @brief Alltoall with variable counts (vector API)
     *
     * Note: Only available for MPI backend (host memory).
     * Returns result as std::vector.
     */
    template<typename T>
    std::vector<T> alltoallv_vec(const std::vector<T>& sendbuf,
                                  const std::vector<int>& sendcounts,
                                  const std::vector<int>& recvcounts);

    // ========================================
    // Simple collectives
    // ========================================

    /**
     * @brief Allgather (all ranks send same count)
     */
    template<typename T>
    RequestPtr allgather(const T* sendbuf, size_t count, T* recvbuf,
                        cudaStream_t stream = nullptr);

    /**
     * @brief Allgather vector API (MPI only)
     */
    template<typename T>
    std::vector<T> allgather_vec(const std::vector<T>& sendbuf);

    /**
     * @brief Allreduce operation
     *
     * @param sendbuf Send buffer
     * @param recvbuf Receive buffer
     * @param count Number of elements
     * @param op Reduction operation (MPI_Op for MPI, ncclRedOp_t for NCCL)
     * @param stream CUDA stream (NCCL only)
     */
    template<typename T>
    RequestPtr allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        int op = 0, cudaStream_t stream = nullptr); // op = MPI_SUM or ncclSum

    // ========================================
    // Backend-specific access
    // ========================================

    /**
     * @brief Get underlying MPI communicator
     * @return Pointer to MPIComm (nullptr if using NCCL backend)
     */
    MPIComm* getMPIComm() { return mpi_comm_.get(); }

    /**
     * @brief Get underlying NCCL communicator
     * @return Pointer to NCCLComm (nullptr if using MPI backend)
     */
    NCCLComm* getNCCLComm() { return nccl_comm_.get(); }

private:
    Backend backend_;
    std::unique_ptr<MPIComm> mpi_comm_;
    std::unique_ptr<NCCLComm> nccl_comm_;

    // Helper to check backend
    void ensureMPI() const {
        if (backend_ != Backend::MPI) {
            throw std::runtime_error("Operation requires MPI backend");
        }
    }

    void ensureNCCL() const {
        if (backend_ != Backend::NCCL) {
            throw std::runtime_error("Operation requires NCCL backend");
        }
    }
};

// ========================================
// Template implementations
// ========================================

template<typename T>
RequestPtr Communicator::send(const T* data, size_t count, int dest, int tag, cudaStream_t stream) {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->send(data, count, dest, tag);
    } else {
        return nccl_comm_->send(data, count, dest, stream);
    }
}

template<typename T>
RequestPtr Communicator::recv(T* data, size_t count, int source, int tag, cudaStream_t stream) {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->recv(data, count, source, tag);
    } else {
        return nccl_comm_->recv(data, count, source, stream);
    }
}

template<typename T>
RequestPtr Communicator::allgatherv(const T* sendbuf, size_t sendcount,
                                    T* recvbuf, const int* recvcounts, const int* displs,
                                    cudaStream_t stream) {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->allgatherv(sendbuf, sendcount, recvbuf, recvcounts, displs);
    } else {
        return nccl_comm_->allgatherv(sendbuf, sendcount, recvbuf, recvcounts, displs, stream);
    }
}

template<typename T>
RequestPtr Communicator::alltoallv(const T* sendbuf, const int* sendcounts, const int* sdispls,
                                   T* recvbuf, const int* recvcounts, const int* rdispls,
                                   cudaStream_t stream) {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->alltoallv(sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls);
    } else {
        return nccl_comm_->alltoallv(sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, stream);
    }
}

template<typename T>
RequestPtr Communicator::allgatherv_auto(const T* sendbuf, size_t sendcount,
                                         T* recvbuf, const int* recvcounts,
                                         cudaStream_t stream) {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->allgatherv_auto(sendbuf, sendcount, recvbuf, recvcounts);
    } else {
        return nccl_comm_->allgatherv_auto(sendbuf, sendcount, recvbuf, recvcounts, stream);
    }
}

template<typename T>
RequestPtr Communicator::alltoallv_auto(const T* sendbuf, const int* sendcounts,
                                        T* recvbuf, const int* recvcounts,
                                        cudaStream_t stream) {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->alltoallv_auto(sendbuf, sendcounts, recvbuf, recvcounts);
    } else {
        return nccl_comm_->alltoallv_auto(sendbuf, sendcounts, recvbuf, recvcounts, stream);
    }
}

template<typename T>
std::vector<T> Communicator::allgatherv_vec(const std::vector<T>& sendbuf,
                                             const std::vector<int>& recvcounts) {
    ensureMPI();
    return mpi_comm_->allgatherv_vec(sendbuf, recvcounts);
}

template<typename T>
std::vector<T> Communicator::alltoallv_vec(const std::vector<T>& sendbuf,
                                            const std::vector<int>& sendcounts,
                                            const std::vector<int>& recvcounts) {
    ensureMPI();
    return mpi_comm_->alltoallv_vec(sendbuf, sendcounts, recvcounts);
}

template<typename T>
RequestPtr Communicator::allgather(const T* sendbuf, size_t count, T* recvbuf,
                                   cudaStream_t stream) {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->allgather(sendbuf, count, recvbuf);
    } else {
        return nccl_comm_->allgather(sendbuf, recvbuf, count, stream);
    }
}

template<typename T>
std::vector<T> Communicator::allgather_vec(const std::vector<T>& sendbuf) {
    ensureMPI();
    return mpi_comm_->allgather_vec(sendbuf);
}

template<typename T>
RequestPtr Communicator::allreduce(const T* sendbuf, T* recvbuf, size_t count,
                                   int op, cudaStream_t stream) {
    if (backend_ == Backend::MPI) {
        MPI_Op mpi_op = (op == 0) ? MPI_SUM : static_cast<MPI_Op>(op);
        return mpi_comm_->allreduce(sendbuf, recvbuf, count, mpi_op);
    } else {
        ncclRedOp_t nccl_op = (op == 0) ? ncclSum : static_cast<ncclRedOp_t>(op);
        return nccl_comm_->allreduce(sendbuf, recvbuf, count, nccl_op, stream);
    }
}

} // namespace stComm

#endif // STCOMM_COMMUNICATOR_H
