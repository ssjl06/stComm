#pragma once

#include "types.h"
#include "request.h"
#include "utils.h"
#include "mpi_check.h"
#include <mpi.h>
#include <vector>
#include <type_traits>
#include <utility>

namespace stComm {

namespace detail {

// Map primitive types to predefined MPI datatypes.
template<typename T> inline MPI_Datatype mpi_datatype() = delete;
template<> inline MPI_Datatype mpi_datatype<char>()           { return MPI_CHAR; }
template<> inline MPI_Datatype mpi_datatype<int8_t>()         { return MPI_INT8_T; }
template<> inline MPI_Datatype mpi_datatype<int16_t>()        { return MPI_INT16_T; }
template<> inline MPI_Datatype mpi_datatype<int32_t>()        { return MPI_INT32_T; }
template<> inline MPI_Datatype mpi_datatype<int64_t>()        { return MPI_INT64_T; }
template<> inline MPI_Datatype mpi_datatype<uint8_t>()        { return MPI_UINT8_T; }
template<> inline MPI_Datatype mpi_datatype<uint16_t>()       { return MPI_UINT16_T; }
template<> inline MPI_Datatype mpi_datatype<uint32_t>()       { return MPI_UINT32_T; }
template<> inline MPI_Datatype mpi_datatype<uint64_t>()       { return MPI_UINT64_T; }
template<> inline MPI_Datatype mpi_datatype<float>()          { return MPI_FLOAT; }
template<> inline MPI_Datatype mpi_datatype<double>()         { return MPI_DOUBLE; }

// Map (T, int) pair types to MPI predefined MAXLOC/MINLOC datatypes.
template<typename T> inline MPI_Datatype mpi_pair_datatype() = delete;
template<> inline MPI_Datatype mpi_pair_datatype<int>()         { return MPI_2INT; }
template<> inline MPI_Datatype mpi_pair_datatype<long>()        { return MPI_LONG_INT; }
template<> inline MPI_Datatype mpi_pair_datatype<float>()       { return MPI_FLOAT_INT; }
template<> inline MPI_Datatype mpi_pair_datatype<double>()      { return MPI_DOUBLE_INT; }
template<> inline MPI_Datatype mpi_pair_datatype<short>()       { return MPI_SHORT_INT; }
template<> inline MPI_Datatype mpi_pair_datatype<long double>() { return MPI_LONG_DOUBLE_INT; }

// MPI's predefined MAXLOC pair types only cover {int, short, long, float, double,
// long double} paired with `int`. For 64-bit integers we cast through `long`
// (LP64: long == int64_t). For unsigned 64-bit we still go through `long` but
// must check the input is non-negative if signedness matters.
template<typename T> struct maxloc_value_cast      { using type = T;    };
template<>           struct maxloc_value_cast<long long>          { using type = long; };
template<>           struct maxloc_value_cast<unsigned long>      { using type = long; };
template<>           struct maxloc_value_cast<unsigned long long> { using type = long; };

} // namespace detail

/**
 * @brief MPI-based communication backend for CPU
 *
 * Implements communication using MPI for host memory.
 * Uses MPI_BYTE to handle arbitrary data sizes efficiently.
 */
class MPIComm {
public:
    MPIComm();
    explicit MPIComm(MPI_Comm comm);
    ~MPIComm() = default;

    // Owns no MPI resource of its own (the communicator is borrowed), but kept
    // non-copyable for parity with NCCLComm and the Comm facade.
    MPIComm(const MPIComm&)            = delete;
    MPIComm& operator=(const MPIComm&) = delete;

    // Static initialization
    static void initialize(int* argc, char*** argv);
    static void finalize();

    // Scalar accessors
    int getRank() const { return rank_; }
    int getSize() const { return size_; }
    Backend getBackend() const { return Backend::MPI; }
    void barrier();

    // Point-to-point communication (async)
    // Automatically handles large data (>2GB) by chunking
    template<typename T>
    std::shared_ptr<MPIRequest> send(const T* data, size_t count, int dest, int tag = 0);

    template<typename T>
    std::shared_ptr<MPIRequest> recv(T* data, size_t count, int source, int tag = 0);

    // Collective communication - with auto displacement
    // Automatically handles large data (>2GB) by chunking
    template<typename T>
    std::shared_ptr<MPIRequest> allgatherv(const T* sendbuf, int sendcount,
                         T* recvbuf, const int* recvcounts);

    template<typename T>
    std::shared_ptr<MPIRequest> alltoallv(const T* sendbuf, const int* sendcounts,
                        T* recvbuf, const int* recvcounts);

    // Broadcast `count` elements of T from `root` to all ranks (async).
    // Automatically handles large data (>2GB) by chunking.
    template<typename T>
    std::shared_ptr<MPIRequest> bcast(T* data, size_t count, int root);

    // Allreduce with MPI_MAXLOC. Pairs the caller's `value` with this rank's id.
    // Returns (max value across ranks, rank that owned that max). Tie-breaking:
    // smaller rank wins (MPI_MAXLOC semantics). Blocking — payload is tiny.
    template<typename T>
    std::pair<T, int> allreduceMaxloc(T value);

    // Exclusive prefix scan over a single value of T. Returns the reduction (op)
    // of values from ranks 0..(myrank-1). For rank 0 the result is the op's
    // identity (zero for MPI_SUM). Blocking — payload is one element.
    template<typename T>
    T exscan(T value, MPI_Op op = MPI_SUM);

    // Async variants of the two reductions above (MPI_Iallreduce / MPI_Iexscan).
    // The scalar result lands in *out once the returned request completes
    // (wait()/test()); the request owns the backing buffers until then. Same
    // shape as NCCLComm's emulated reductions so the Comm facade routes both by
    // Space tag.
    template<typename T>
    std::shared_ptr<MPIRequest> allreduceMaxloc(T value, std::pair<T, int>* out);

    template<typename T>
    std::shared_ptr<MPIRequest> exscan(T value, T* out, MPI_Op op = MPI_SUM);

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
std::shared_ptr<MPIRequest> MPIComm::send(const T* data, size_t count, int dest, int tag) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    const size_t total_bytes = count * sizeof(T);
    const size_t MAX_CHUNK_SIZE = 1073741824;  // 1GB per chunk

    // Check if we need chunking for large data (>2GB)
    if (total_bytes <= static_cast<size_t>(INT_MAX)) {
        // Small data: single MPI call
        auto req = std::make_shared<MPIRequest>();
        STCOMM_MPI_CHECK(MPI_Isend(data, total_bytes, MPI_BYTE, dest, tag, comm_, &req->getHandle()));
        return req;
    }

    // Large data: split into chunks, all tracked by one MPIRequest.
    auto req = std::make_shared<MPIRequest>();
    const char* byte_data = reinterpret_cast<const char*>(data);
    size_t offset = 0;

    while (offset < total_bytes) {
        size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_bytes - offset);
        STCOMM_MPI_CHECK(MPI_Isend(byte_data + offset, static_cast<int>(chunk_size), MPI_BYTE, dest, tag, comm_, &req->addHandle()));
        offset += chunk_size;
    }

    return req;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::recv(T* data, size_t count, int source, int tag) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    const size_t total_bytes = count * sizeof(T);
    const size_t MAX_CHUNK_SIZE = 1073741824;  // 1GB per chunk

    // Check if we need chunking for large data (>2GB)
    if (total_bytes <= static_cast<size_t>(INT_MAX)) {
        // Small data: single MPI call
        auto req = std::make_shared<MPIRequest>();
        STCOMM_MPI_CHECK(MPI_Irecv(data, total_bytes, MPI_BYTE, source, tag, comm_, &req->getHandle()));
        return req;
    }

    // Large data: split into chunks, all tracked by one MPIRequest.
    auto req = std::make_shared<MPIRequest>();
    char* byte_data = reinterpret_cast<char*>(data);
    size_t offset = 0;

    while (offset < total_bytes) {
        size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_bytes - offset);
        STCOMM_MPI_CHECK(MPI_Irecv(byte_data + offset, static_cast<int>(chunk_size), MPI_BYTE, source, tag, comm_, &req->addHandle()));
        offset += chunk_size;
    }

    return req;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::allgatherv(const T* sendbuf, int sendcount,
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

    STCOMM_MPI_CHECK(MPI_Iallgatherv(sendbuf, sendcount * sizeof(T), MPI_BYTE,
                    recvbuf, byte_recvcounts.data(), byte_displs.data(), MPI_BYTE,
                    comm_, &req->getHandle()));

    return req;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::alltoallv(const T* sendbuf, const int* sendcounts,
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

    STCOMM_MPI_CHECK(MPI_Ialltoallv(sendbuf, byte_sendcounts.data(), byte_sdispls.data(), MPI_BYTE,
                   recvbuf, byte_recvcounts.data(), byte_rdispls.data(), MPI_BYTE,
                   comm_, &req->getHandle()));

    return req;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::bcast(T* data, size_t count, int root) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    const size_t total_bytes = count * sizeof(T);
    const size_t MAX_CHUNK_SIZE = 1073741824;  // 1GB per chunk

    if (total_bytes <= static_cast<size_t>(INT_MAX)) {
        auto req = std::make_shared<MPIRequest>();
        STCOMM_MPI_CHECK(MPI_Ibcast(data, static_cast<int>(total_bytes), MPI_BYTE, root,
                   comm_, &req->getHandle()));
        return req;
    }

    auto req = std::make_shared<MPIRequest>();
    char* byte_data = reinterpret_cast<char*>(data);
    size_t offset = 0;
    while (offset < total_bytes) {
        size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_bytes - offset);
        STCOMM_MPI_CHECK(MPI_Ibcast(byte_data + offset, static_cast<int>(chunk_size), MPI_BYTE,
                   root, comm_, &req->addHandle()));
        offset += chunk_size;
    }
    return req;
}

template<typename T>
std::pair<T, int> MPIComm::allreduceMaxloc(T value) {
    // Pack (value cast to a MPI-supported pair value type, rank).
    using MPIValue = typename detail::maxloc_value_cast<T>::type;
    struct Pair { MPIValue value; int rank; };
    Pair local{static_cast<MPIValue>(value), rank_};
    Pair global{};
    STCOMM_MPI_CHECK(MPI_Allreduce(&local, &global, 1,
                  detail::mpi_pair_datatype<MPIValue>(),
                  MPI_MAXLOC, comm_));
    return { static_cast<T>(global.value), global.rank };
}

template<typename T>
T MPIComm::exscan(T value, MPI_Op op) {
    T result = T{};
    STCOMM_MPI_CHECK(MPI_Exscan(&value, &result, 1, detail::mpi_datatype<T>(), op, comm_));
    // MPI_Exscan leaves rank 0's recvbuf undefined; return identity (zero) instead.
    return rank_ == 0 ? T{} : result;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::allreduceMaxloc(T value, std::pair<T, int>* out) {
    using MPIValue = typename detail::maxloc_value_cast<T>::type;
    struct Pair { MPIValue value; int rank; };
    // Buffers must outlive the non-blocking call; the request owns them.
    struct Buf { Pair local; Pair global; };
    auto buf = std::make_shared<Buf>();
    buf->local = Pair{static_cast<MPIValue>(value), rank_};

    auto req = std::make_shared<MPIRequest>();
    STCOMM_MPI_CHECK(MPI_Iallreduce(&buf->local, &buf->global, 1,
                   detail::mpi_pair_datatype<MPIValue>(), MPI_MAXLOC,
                   comm_, &req->addHandle()));
    req->setScratch(buf);
    req->setFinalizer([buf, out]() {
        *out = { static_cast<T>(buf->global.value), buf->global.rank };
    });
    return req;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::exscan(T value, T* out, MPI_Op op) {
    struct Buf { T send; T recv; };
    auto buf = std::make_shared<Buf>();
    buf->send = value;

    auto req = std::make_shared<MPIRequest>();
    STCOMM_MPI_CHECK(MPI_Iexscan(&buf->send, &buf->recv, 1, detail::mpi_datatype<T>(), op,
                comm_, &req->addHandle()));
    const int rank = rank_;
    req->setScratch(buf);
    req->setFinalizer([buf, out, rank]() {
        // MPI_Iexscan leaves rank 0's recvbuf undefined; return identity.
        *out = (rank == 0) ? T{} : buf->recv;
    });
    return req;
}

} // namespace stComm
