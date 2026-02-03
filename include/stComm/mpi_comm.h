#pragma once

#include "comm_base.h"
#include "utils.h"
#include <mpi.h>
#include <vector>
#include <type_traits>

namespace stComm {

/**
 * @brief MPI-based communication backend for CPU
 *
 * Implements communication using MPI for host memory.
 * Uses MPI_BYTE to handle arbitrary data sizes efficiently.
 */
class MPIComm : public CommBase {
public:
    MPIComm();
    explicit MPIComm(MPI_Comm comm);
    ~MPIComm() override = default;

    // Static initialization
    static void initialize(int* argc, char*** argv);
    static void finalize();

    // CommBase interface
    int getRank() const override { return rank_; }
    int getSize() const override { return size_; }
    Backend getBackend() const override { return Backend::MPI; }
    void barrier() override;

    // Point-to-point communication (async)
    // Automatically handles large data (>2GB) by chunking
    template<typename T>
    RequestPtr send(const T* data, size_t count, int dest, int tag = 0);

    template<typename T>
    RequestPtr recv(T* data, size_t count, int source, int tag = 0);

    // Collective communication - with auto displacement
    // Automatically handles large data (>2GB) by chunking
    template<typename T>
    RequestPtr allgatherv(const T* sendbuf, int sendcount,
                         T* recvbuf, const int* recvcounts);

    template<typename T>
    RequestPtr alltoallv(const T* sendbuf, const int* sendcounts,
                        T* recvbuf, const int* recvcounts);

    // Get native handle
    MPI_Comm getHandle() const { return comm_; }

private:
    MPI_Comm comm_;
    int rank_;
    int size_;
};

// ============================================================================
// Template implementations
// ============================================================================

template<typename T>
RequestPtr MPIComm::send(const T* data, size_t count, int dest, int tag) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    const size_t total_bytes = count * sizeof(T);
    const size_t MAX_CHUNK_SIZE = 1073741824;  // 1GB per chunk

    // Check if we need chunking for large data (>2GB)
    if (total_bytes <= static_cast<size_t>(INT_MAX)) {
        // Small data: single MPI call
        auto req = std::make_shared<MPIRequest>();
        MPI_Isend(data, total_bytes, MPI_BYTE, dest, tag, comm_, &req->getHandle());
        return req;
    }

    // Large data: split into chunks
    auto multi_req = std::make_shared<MultiMPIRequest>();
    const char* byte_data = reinterpret_cast<const char*>(data);
    size_t offset = 0;

    while (offset < total_bytes) {
        size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_bytes - offset);
        MPI_Request mpi_req;
        MPI_Isend(byte_data + offset, static_cast<int>(chunk_size), MPI_BYTE, dest, tag, comm_, &mpi_req);
        multi_req->addRequest(mpi_req);
        offset += chunk_size;
    }

    return multi_req;
}

template<typename T>
RequestPtr MPIComm::recv(T* data, size_t count, int source, int tag) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    const size_t total_bytes = count * sizeof(T);
    const size_t MAX_CHUNK_SIZE = 1073741824;  // 1GB per chunk

    // Check if we need chunking for large data (>2GB)
    if (total_bytes <= static_cast<size_t>(INT_MAX)) {
        // Small data: single MPI call
        auto req = std::make_shared<MPIRequest>();
        MPI_Irecv(data, total_bytes, MPI_BYTE, source, tag, comm_, &req->getHandle());
        return req;
    }

    // Large data: split into chunks
    auto multi_req = std::make_shared<MultiMPIRequest>();
    char* byte_data = reinterpret_cast<char*>(data);
    size_t offset = 0;

    while (offset < total_bytes) {
        size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_bytes - offset);
        MPI_Request mpi_req;
        MPI_Irecv(byte_data + offset, static_cast<int>(chunk_size), MPI_BYTE, source, tag, comm_, &mpi_req);
        multi_req->addRequest(mpi_req);
        offset += chunk_size;
    }

    return multi_req;
}

template<typename T>
RequestPtr MPIComm::allgatherv(const T* sendbuf, int sendcount,
                               T* recvbuf, const int* recvcounts) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();

    // Auto-calculate displacements
    auto displs = Utils::calculateDisplacements(recvcounts, size_);

    // Convert counts and displacements to bytes
    std::vector<int> byte_recvcounts(size_);
    std::vector<int> byte_displs(size_);

    for (int i = 0; i < size_; ++i) {
        byte_recvcounts[i] = recvcounts[i] * sizeof(T);
        byte_displs[i] = displs[i] * sizeof(T);
    }

    MPI_Iallgatherv(sendbuf, sendcount * sizeof(T), MPI_BYTE,
                    recvbuf, byte_recvcounts.data(), byte_displs.data(), MPI_BYTE,
                    comm_, &req->getHandle());

    return req;
}

template<typename T>
RequestPtr MPIComm::alltoallv(const T* sendbuf, const int* sendcounts,
                              T* recvbuf, const int* recvcounts) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();

    // Auto-calculate displacements for both send and recv
    auto sdispls = Utils::calculateDisplacements(sendcounts, size_);
    auto rdispls = Utils::calculateDisplacements(recvcounts, size_);

    // Convert counts and displacements to bytes
    std::vector<int> byte_sendcounts(size_);
    std::vector<int> byte_sdispls(size_);
    std::vector<int> byte_recvcounts(size_);
    std::vector<int> byte_rdispls(size_);

    for (int i = 0; i < size_; ++i) {
        byte_sendcounts[i] = sendcounts[i] * sizeof(T);
        byte_sdispls[i] = sdispls[i] * sizeof(T);
        byte_recvcounts[i] = recvcounts[i] * sizeof(T);
        byte_rdispls[i] = rdispls[i] * sizeof(T);
    }

    MPI_Ialltoallv(sendbuf, byte_sendcounts.data(), byte_sdispls.data(), MPI_BYTE,
                   recvbuf, byte_recvcounts.data(), byte_rdispls.data(), MPI_BYTE,
                   comm_, &req->getHandle());

    return req;
}

} // namespace stComm
