#ifndef STCOMM_MPI_COMM_H
#define STCOMM_MPI_COMM_H

#include "types.h"
#include "request.h"
#include "utils.h"
#include <mpi.h>
#include <vector>
#include <type_traits>

namespace stComm {

// MPI datatype mapper for primitive types
template<typename T>
struct MPITypeMap {
    static MPI_Datatype type();
};

// Specializations for common types
template<> inline MPI_Datatype MPITypeMap<char>::type() { return MPI_CHAR; }
template<> inline MPI_Datatype MPITypeMap<int>::type() { return MPI_INT; }
template<> inline MPI_Datatype MPITypeMap<long>::type() { return MPI_LONG; }
template<> inline MPI_Datatype MPITypeMap<long long>::type() { return MPI_LONG_LONG; }
template<> inline MPI_Datatype MPITypeMap<unsigned char>::type() { return MPI_UNSIGNED_CHAR; }
template<> inline MPI_Datatype MPITypeMap<unsigned int>::type() { return MPI_UNSIGNED; }
template<> inline MPI_Datatype MPITypeMap<unsigned long>::type() { return MPI_UNSIGNED_LONG; }
template<> inline MPI_Datatype MPITypeMap<float>::type() { return MPI_FLOAT; }
template<> inline MPI_Datatype MPITypeMap<double>::type() { return MPI_DOUBLE; }
template<> inline MPI_Datatype MPITypeMap<int8_t>::type() { return MPI_INT8_T; }
template<> inline MPI_Datatype MPITypeMap<int16_t>::type() { return MPI_INT16_T; }
template<> inline MPI_Datatype MPITypeMap<int32_t>::type() { return MPI_INT32_T; }
template<> inline MPI_Datatype MPITypeMap<int64_t>::type() { return MPI_INT64_T; }
template<> inline MPI_Datatype MPITypeMap<uint8_t>::type() { return MPI_UINT8_T; }
template<> inline MPI_Datatype MPITypeMap<uint16_t>::type() { return MPI_UINT16_T; }
template<> inline MPI_Datatype MPITypeMap<uint32_t>::type() { return MPI_UINT32_T; }
template<> inline MPI_Datatype MPITypeMap<uint64_t>::type() { return MPI_UINT64_T; }

class MPIComm {
public:
    MPIComm();
    explicit MPIComm(MPI_Comm comm);
    ~MPIComm() = default;

    // Initialize MPI (call once at program start)
    static void initialize(int* argc, char*** argv);

    // Finalize MPI (call once at program end)
    static void finalize();

    // Get rank and size
    int getRank() const;
    int getSize() const;

    // Point-to-point communication (async)
    template<typename T>
    RequestPtr send(const T* data, size_t count, int dest, int tag = 0);

    template<typename T>
    RequestPtr recv(T* data, size_t count, int source, int tag = 0);

    // Collective communication (async) - Low-level API
    template<typename T>
    RequestPtr allgatherv(const T* sendbuf, size_t sendcount,
                         T* recvbuf, const int* recvcounts, const int* displs);

    template<typename T>
    RequestPtr alltoallv(const T* sendbuf, const int* sendcounts, const int* sdispls,
                        T* recvbuf, const int* recvcounts, const int* rdispls);

    // High-level API - Auto displacement calculation
    template<typename T>
    RequestPtr allgatherv_auto(const T* sendbuf, size_t sendcount,
                               T* recvbuf, const int* recvcounts);

    template<typename T>
    RequestPtr alltoallv_auto(const T* sendbuf, const int* sendcounts,
                              T* recvbuf, const int* recvcounts);

    // Vector-based API for maximum convenience
    template<typename T>
    std::vector<T> allgatherv_vec(const std::vector<T>& sendbuf, const std::vector<int>& recvcounts);

    template<typename T>
    std::vector<T> alltoallv_vec(const std::vector<T>& sendbuf,
                                  const std::vector<int>& sendcounts,
                                  const std::vector<int>& recvcounts);

    // Simple allgather (all ranks send same count)
    template<typename T>
    RequestPtr allgather(const T* sendbuf, size_t count, T* recvbuf);

    template<typename T>
    std::vector<T> allgather_vec(const std::vector<T>& sendbuf);

    // Allreduce operations
    template<typename T>
    RequestPtr allreduce(const T* sendbuf, T* recvbuf, size_t count, MPI_Op op = MPI_SUM);

    // Barrier
    void barrier();

    MPI_Comm getHandle() const { return comm_; }

private:
    MPI_Comm comm_;
    int rank_;
    int size_;
};

// Template implementations
template<typename T>
RequestPtr MPIComm::send(const T* data, size_t count, int dest, int tag) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();
    MPI_Datatype mpi_type = MPITypeMap<T>::type();

    // Use MPI_BYTE for better handling of large data
    MPI_Isend(data, count * sizeof(T), MPI_BYTE, dest, tag, comm_, &req->getHandle());

    return req;
}

template<typename T>
RequestPtr MPIComm::recv(T* data, size_t count, int source, int tag) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();
    MPI_Datatype mpi_type = MPITypeMap<T>::type();

    // Use MPI_BYTE for better handling of large data
    MPI_Irecv(data, count * sizeof(T), MPI_BYTE, source, tag, comm_, &req->getHandle());

    return req;
}

template<typename T>
RequestPtr MPIComm::allgatherv(const T* sendbuf, size_t sendcount,
                               T* recvbuf, const int* recvcounts, const int* displs) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();

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
RequestPtr MPIComm::alltoallv(const T* sendbuf, const int* sendcounts, const int* sdispls,
                              T* recvbuf, const int* recvcounts, const int* rdispls) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();

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

// High-level API implementations

template<typename T>
RequestPtr MPIComm::allgatherv_auto(const T* sendbuf, size_t sendcount,
                                     T* recvbuf, const int* recvcounts) {
    // Automatically calculate displacements from counts
    auto displs = Utils::calculate_displacements(recvcounts, size_);
    return allgatherv(sendbuf, sendcount, recvbuf, recvcounts, displs.data());
}

template<typename T>
RequestPtr MPIComm::alltoallv_auto(const T* sendbuf, const int* sendcounts,
                                    T* recvbuf, const int* recvcounts) {
    // Automatically calculate displacements for both send and recv
    auto sdispls = Utils::calculate_displacements(sendcounts, size_);
    auto rdispls = Utils::calculate_displacements(recvcounts, size_);
    return alltoallv(sendbuf, sendcounts, sdispls.data(),
                     recvbuf, recvcounts, rdispls.data());
}

template<typename T>
std::vector<T> MPIComm::allgatherv_vec(const std::vector<T>& sendbuf,
                                        const std::vector<int>& recvcounts) {
    if (recvcounts.size() != static_cast<size_t>(size_)) {
        throw std::runtime_error("recvcounts size must match MPI size");
    }

    // Calculate total receive buffer size and allocate
    int total_recv = Utils::total_size(recvcounts);
    std::vector<T> recvbuf(total_recv);

    // Perform allgatherv with auto displacement
    auto req = allgatherv_auto(sendbuf.data(), sendbuf.size(),
                               recvbuf.data(), recvcounts.data());
    req->wait();

    return recvbuf;
}

template<typename T>
std::vector<T> MPIComm::alltoallv_vec(const std::vector<T>& sendbuf,
                                       const std::vector<int>& sendcounts,
                                       const std::vector<int>& recvcounts) {
    if (sendcounts.size() != static_cast<size_t>(size_) ||
        recvcounts.size() != static_cast<size_t>(size_)) {
        throw std::runtime_error("counts size must match MPI size");
    }

    // Verify sendbuf size matches sendcounts
    int expected_send = Utils::total_size(sendcounts);
    if (sendbuf.size() != static_cast<size_t>(expected_send)) {
        throw std::runtime_error("sendbuf size does not match sendcounts");
    }

    // Calculate total receive buffer size and allocate
    int total_recv = Utils::total_size(recvcounts);
    std::vector<T> recvbuf(total_recv);

    // Perform alltoallv with auto displacement
    auto req = alltoallv_auto(sendbuf.data(), sendcounts.data(),
                              recvbuf.data(), recvcounts.data());
    req->wait();

    return recvbuf;
}

template<typename T>
RequestPtr MPIComm::allgather(const T* sendbuf, size_t count, T* recvbuf) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();

    MPI_Iallgather(sendbuf, count * sizeof(T), MPI_BYTE,
                   recvbuf, count * sizeof(T), MPI_BYTE,
                   comm_, &req->getHandle());

    return req;
}

template<typename T>
std::vector<T> MPIComm::allgather_vec(const std::vector<T>& sendbuf) {
    std::vector<T> recvbuf(sendbuf.size() * size_);
    auto req = allgather(sendbuf.data(), sendbuf.size(), recvbuf.data());
    req->wait();
    return recvbuf;
}

template<typename T>
RequestPtr MPIComm::allreduce(const T* sendbuf, T* recvbuf, size_t count, MPI_Op op) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();
    MPI_Datatype mpi_type = MPITypeMap<T>::type();

    MPI_Iallreduce(sendbuf, recvbuf, count, mpi_type, op, comm_, &req->getHandle());

    return req;
}

} // namespace stComm

#endif // STCOMM_MPI_COMM_H
