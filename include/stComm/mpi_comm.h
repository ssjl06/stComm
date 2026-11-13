#pragma once

#include "types.h"
#include "request.h"
#include "utils.h"
#include "mpi_check.h"
#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
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
// long double} paired with `int`, so 64-bit integers travel as `long` (LP64:
// long == int64_t). Unsigned 64-bit values cannot simply be cast — anything
// >= 2^63 would turn negative and lose the comparison — so they are biased by
// 2^63 (top bit flipped), which maps the unsigned order onto the signed order
// exactly; maxloc_decode undoes it.
template<typename T> struct maxloc_value_cast      { using type = T;    };
template<>           struct maxloc_value_cast<long long>          { using type = long; };
template<>           struct maxloc_value_cast<unsigned long>      { using type = long; };
template<>           struct maxloc_value_cast<unsigned long long> { using type = long; };

template<typename T>
constexpr bool maxloc_is_biased =
    std::is_integral<T>::value && std::is_unsigned<T>::value &&
    !std::is_same<typename maxloc_value_cast<T>::type, T>::value;

constexpr unsigned long kMaxlocBias = 1UL << 63;

template<typename T>
inline typename maxloc_value_cast<T>::type maxloc_encode(T v) {
    using M = typename maxloc_value_cast<T>::type;
    if constexpr (maxloc_is_biased<T>) {
        static_assert(sizeof(T) == 8 && sizeof(long) == 8, "LP64 assumed");
        return static_cast<M>(static_cast<unsigned long>(v) ^ kMaxlocBias);
    } else {
        static_assert(sizeof(M) >= sizeof(T), "MAXLOC wire type narrower than T");
        return static_cast<M>(v);
    }
}

template<typename T>
inline T maxloc_decode(typename maxloc_value_cast<T>::type m) {
    if constexpr (maxloc_is_biased<T>) {
        return static_cast<T>(static_cast<unsigned long>(m) ^ kMaxlocBias);
    } else {
        return static_cast<T>(m);
    }
}

// The v-collectives describe one element as a contiguous run of sizeof(T)
// bytes, so MPI's int counts and displacements are in *elements*: the limit is
// INT_MAX elements per rank-side (16 GiB of 8-byte values), not INT_MAX bytes.
// Freed on scope exit — MPI keeps a datatype alive for operations still using
// it, so the non-blocking call may outlive this object.
class ElemType {
public:
    explicit ElemType(std::size_t elem_size) {
        if (elem_size == 1) return;  // MPI_BYTE itself
        if (elem_size > static_cast<std::size_t>(INT_MAX)) {
            throw std::overflow_error("stComm MPI: element size exceeds INT_MAX bytes");
        }
        STCOMM_MPI_CHECK(MPI_Type_contiguous(static_cast<int>(elem_size), MPI_BYTE, &type_));
        owned_ = true;
        STCOMM_MPI_CHECK(MPI_Type_commit(&type_));
    }
    ~ElemType() {
        if (owned_) MPI_Type_free(&type_);  // must not throw
    }
    ElemType(const ElemType&)            = delete;
    ElemType& operator=(const ElemType&) = delete;

    MPI_Datatype get() const { return type_; }

private:
    MPI_Datatype type_  = MPI_BYTE;
    bool         owned_ = false;
};

// Per-rank element counts + displacements for one side of a v-collective. MPI
// may read these arrays until the non-blocking call completes, so they live in
// the request's scratch (MPIRequest::setScratch), not on the caller's stack.
struct ElemLayout {
    std::vector<int> counts;
    std::vector<int> displs;
};

// Offsets are accumulated in int64 and refused past INT_MAX: an int running
// sum would wrap into a negative displacement with no error.
inline ElemLayout elemLayout(const int* counts, int nranks, const char* op) {
    const auto n = static_cast<std::size_t>(nranks);
    ElemLayout l;
    l.counts.assign(counts, counts + n);
    l.displs.resize(n);
    std::int64_t offset = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (counts[i] < 0 || offset > INT_MAX) {
            throw std::overflow_error(
                std::string("stComm MPI ") + op + ": rank " + std::to_string(i) +
                " has count " + std::to_string(counts[i]) + " at element offset " +
                std::to_string(offset) + " — counts must be >= 0 and offsets "
                "must fit in int (max INT_MAX elements per side)");
        }
        l.displs[i] = static_cast<int>(offset);
        offset += counts[i];
    }
    return l;
}

// Chunk size, in bytes, of the point-to-point messages behind the size_t-count
// collectives. Every MPI count stays far below INT_MAX whatever the payload.
// A reference so tests can shrink it to exercise multi-chunk transfers with
// small data; every rank must use the same value.
inline std::size_t& p2pChunkBytes() {
    static std::size_t bytes = std::size_t{1} << 30;  // 1 GiB
    return bytes;
}

// Tag reserved for that point-to-point traffic (within MPI's guaranteed tag
// range). It shares the communicator with user send()/recv(), which therefore
// refuse it: a user message under this tag could be matched by a collective's
// chunk receive, or swallow one of its chunks.
constexpr int kLargeCollectiveTag = 32765;

inline void rejectReservedTag(int tag) {
    if (tag == kLargeCollectiveTag) {
        throw std::invalid_argument("stComm MPI: tag " + std::to_string(tag) +
                                    " is reserved for the size_t-count collectives");
    }
}

} // namespace detail

/**
 * @brief MPI-based communication backend for CPU
 *
 * Implements communication using MPI for host memory. Data movers ship raw
 * bytes (any trivially copyable T); reductions use the real MPI datatype.
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

    // Collective communication - with auto displacement. Counts are int
    // elements; each side may total up to INT_MAX elements (not bytes), and a
    // layout past that throws std::overflow_error before any MPI call. Every
    // rank must be given a layout that fits, or the collective desynchronises
    // — for payloads that may not fit, use the size_t variants below.
    template<typename T>
    std::shared_ptr<MPIRequest> allgatherv(const T* sendbuf, int sendcount,
                         T* recvbuf, const int* recvcounts);

    template<typename T>
    std::shared_ptr<MPIRequest> alltoallv(const T* sendbuf, const int* sendcounts,
                        T* recvbuf, const int* recvcounts);

    // Unlimited-size variants: size_t counts, no INT_MAX limit of any kind
    // (per pair, per rank or in total). The payload travels as point-to-point
    // messages chunked at 1 GiB — each pair's count is known to both ends, so
    // no rank has to learn anything about the others' layouts, and submission
    // stays fully non-blocking. Same contract as the int versions otherwise:
    // every rank calls together, auto displacement, and this rank's send count
    // to itself must equal its receive count from itself.
    //
    // A given exchange must use the SAME variant on every rank — the int
    // versions are MPI collectives and do not interoperate with these.
    template<typename T>
    std::shared_ptr<MPIRequest> allgatherv(const T* sendbuf, std::size_t sendcount,
                         T* recvbuf, const std::size_t* recvcounts);

    template<typename T>
    std::shared_ptr<MPIRequest> alltoallv(const T* sendbuf, const std::size_t* sendcounts,
                        T* recvbuf, const std::size_t* recvcounts);

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

    // Element-wise allreduce over `count` values of T (async): recvbuf receives
    // the reduction (op) of every rank's sendbuf. Unlike the byte-based data
    // movers above, a reduction needs the real datatype so MPI applies `op`
    // element-wise. Caller-owned buffers must outlive the returned request.
    // Chunked so each MPI call stays under INT_MAX elements (reductions are
    // element-wise, so chunking is safe). In-place is allowed
    // (sendbuf == recvbuf, mapped to MPI_IN_PLACE).
    template<typename T>
    std::shared_ptr<MPIRequest> allreduce(const T* sendbuf, T* recvbuf,
                                          size_t count, MPI_Op op = MPI_SUM);

    // Get native handle
    MPI_Comm getHandle() const { return comm_; }

private:
    // Issue `nbytes` to/from one peer as a run of chunked non-blocking
    // messages, all tracked by `req`. Callers skip empty transfers (both ends
    // know the count), so at least one message is always issued.
    void isendBytes(MPIRequest& req, const void* data, std::size_t nbytes, int dest);
    void irecvBytes(MPIRequest& req, void* data, std::size_t nbytes, int source);

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
    detail::rejectReservedTag(tag);

    const size_t total_bytes = count * sizeof(T);
    const size_t MAX_CHUNK_SIZE = 1073741824;  // 1GB per chunk

    // Check if we need chunking for large data (>2GB)
    if (total_bytes <= static_cast<size_t>(INT_MAX)) {
        // Small data: single MPI call
        auto req = std::make_shared<MPIRequest>();
        STCOMM_MPI_CHECK(MPI_Isend(data, static_cast<int>(total_bytes), MPI_BYTE, dest, tag, comm_, &req->getHandle()));
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
    detail::rejectReservedTag(tag);

    const size_t total_bytes = count * sizeof(T);
    const size_t MAX_CHUNK_SIZE = 1073741824;  // 1GB per chunk

    // Check if we need chunking for large data (>2GB)
    if (total_bytes <= static_cast<size_t>(INT_MAX)) {
        // Small data: single MPI call
        auto req = std::make_shared<MPIRequest>();
        STCOMM_MPI_CHECK(MPI_Irecv(data, static_cast<int>(total_bytes), MPI_BYTE, source, tag, comm_, &req->addRecvHandle()));
        return req;
    }

    // Large data: split into chunks, all tracked by one MPIRequest.
    auto req = std::make_shared<MPIRequest>();
    char* byte_data = reinterpret_cast<char*>(data);
    size_t offset = 0;

    while (offset < total_bytes) {
        size_t chunk_size = std::min(MAX_CHUNK_SIZE, total_bytes - offset);
        STCOMM_MPI_CHECK(MPI_Irecv(byte_data + offset, static_cast<int>(chunk_size), MPI_BYTE, source, tag, comm_, &req->addRecvHandle()));
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

    if (sendcount < 0) {
        throw std::overflow_error("stComm MPI allgatherv: negative sendcount");
    }
    auto recv = std::make_shared<detail::ElemLayout>(
        detail::elemLayout(recvcounts, size_, "allgatherv"));
    const detail::ElemType elem(sizeof(T));

    STCOMM_MPI_CHECK(MPI_Iallgatherv(sendbuf, sendcount, elem.get(),
                    recvbuf, recv->counts.data(), recv->displs.data(), elem.get(),
                    comm_, &req->getHandle()));
    req->setScratch(recv);  // arrays must outlive the non-blocking call

    return req;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::alltoallv(const T* sendbuf, const int* sendcounts,
                              T* recvbuf, const int* recvcounts) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    auto req = std::make_shared<MPIRequest>();

    // Element counts/displacements for both sides (auto displacement),
    // range-checked.
    using Layouts = std::pair<detail::ElemLayout, detail::ElemLayout>;
    auto layouts = std::make_shared<Layouts>(
        detail::elemLayout(sendcounts, size_, "alltoallv"),
        detail::elemLayout(recvcounts, size_, "alltoallv"));
    const detail::ElemLayout& snd = layouts->first;
    const detail::ElemLayout& rcv = layouts->second;
    const detail::ElemType elem(sizeof(T));

    STCOMM_MPI_CHECK(MPI_Ialltoallv(sendbuf, snd.counts.data(), snd.displs.data(), elem.get(),
                   recvbuf, rcv.counts.data(), rcv.displs.data(), elem.get(),
                   comm_, &req->getHandle()));
    req->setScratch(layouts);  // arrays must outlive the non-blocking call

    return req;
}

inline void MPIComm::isendBytes(MPIRequest& req, const void* data, std::size_t nbytes,
                                int dest) {
    const std::size_t chunk_max =
        std::min(detail::p2pChunkBytes(), static_cast<std::size_t>(INT_MAX));
    const char* bytes = static_cast<const char*>(data);
    for (std::size_t off = 0; off < nbytes;) {
        const std::size_t chunk = std::min(chunk_max, nbytes - off);
        STCOMM_MPI_CHECK(MPI_Isend(bytes + off, static_cast<int>(chunk), MPI_BYTE, dest,
                                   detail::kLargeCollectiveTag, comm_, &req.addHandle()));
        off += chunk;
    }
}

inline void MPIComm::irecvBytes(MPIRequest& req, void* data, std::size_t nbytes,
                                int source) {
    const std::size_t chunk_max =
        std::min(detail::p2pChunkBytes(), static_cast<std::size_t>(INT_MAX));
    char* bytes = static_cast<char*>(data);
    for (std::size_t off = 0; off < nbytes;) {
        const std::size_t chunk = std::min(chunk_max, nbytes - off);
        STCOMM_MPI_CHECK(MPI_Irecv(bytes + off, static_cast<int>(chunk), MPI_BYTE, source,
                                   detail::kLargeCollectiveTag, comm_, &req.addRecvHandle()));
        off += chunk;
    }
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::allgatherv(const T* sendbuf, std::size_t sendcount,
                               T* recvbuf, const std::size_t* recvcounts) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    const auto nranks = static_cast<std::size_t>(size_);
    const auto me     = static_cast<std::size_t>(rank_);
    if (recvcounts[me] != sendcount) {
        throw std::invalid_argument(
            "stComm MPI allgatherv: this rank sends " + std::to_string(sendcount) +
            " elements but expects " + std::to_string(recvcounts[me]) + " from itself");
    }

    auto req = std::make_shared<MPIRequest>();
    // Receives are posted before any send so no chunk ever waits unexpected.
    T* mine = nullptr;
    std::size_t offset = 0;  // in elements
    for (std::size_t r = 0; r < nranks; ++r) {
        T* dst = recvbuf + offset;
        if (r == me) {
            mine = dst;
        } else if (recvcounts[r] > 0) {
            irecvBytes(*req, dst, recvcounts[r] * sizeof(T), static_cast<int>(r));
        }
        offset += recvcounts[r];
    }
    if (sendcount > 0) {
        for (std::size_t r = 0; r < nranks; ++r) {
            if (r != me) isendBytes(*req, sendbuf, sendcount * sizeof(T), static_cast<int>(r));
        }
        if (mine != sendbuf) std::memmove(mine, sendbuf, sendcount * sizeof(T));
    }
    return req;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::alltoallv(const T* sendbuf, const std::size_t* sendcounts,
                              T* recvbuf, const std::size_t* recvcounts) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    const auto nranks = static_cast<std::size_t>(size_);
    const auto me     = static_cast<std::size_t>(rank_);
    if (sendcounts[me] != recvcounts[me]) {
        throw std::invalid_argument(
            "stComm MPI alltoallv: this rank sends itself " + std::to_string(sendcounts[me]) +
            " elements but expects " + std::to_string(recvcounts[me]));
    }

    auto req = std::make_shared<MPIRequest>();
    // Receives are posted before any send so no chunk ever waits unexpected.
    T* self_dst = nullptr;
    std::size_t offset = 0;  // in elements
    for (std::size_t r = 0; r < nranks; ++r) {
        T* dst = recvbuf + offset;
        if (r == me) {
            self_dst = dst;
        } else if (recvcounts[r] > 0) {
            irecvBytes(*req, dst, recvcounts[r] * sizeof(T), static_cast<int>(r));
        }
        offset += recvcounts[r];
    }
    offset = 0;
    for (std::size_t r = 0; r < nranks; ++r) {
        const T* src = sendbuf + offset;
        if (r == me) {
            if (sendcounts[r] > 0 && self_dst != src) {
                std::memmove(self_dst, src, sendcounts[r] * sizeof(T));
            }
        } else if (sendcounts[r] > 0) {
            isendBytes(*req, src, sendcounts[r] * sizeof(T), static_cast<int>(r));
        }
        offset += sendcounts[r];
    }
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
    Pair local{detail::maxloc_encode<T>(value), rank_};
    Pair global{};
    STCOMM_MPI_CHECK(MPI_Allreduce(&local, &global, 1,
                  detail::mpi_pair_datatype<MPIValue>(),
                  MPI_MAXLOC, comm_));
    return { detail::maxloc_decode<T>(global.value), global.rank };
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
    buf->local = Pair{detail::maxloc_encode<T>(value), rank_};

    auto req = std::make_shared<MPIRequest>();
    STCOMM_MPI_CHECK(MPI_Iallreduce(&buf->local, &buf->global, 1,
                   detail::mpi_pair_datatype<MPIValue>(), MPI_MAXLOC,
                   comm_, &req->addHandle()));
    req->setScratch(buf);
    req->setFinalizer([buf, out]() {
        *out = { detail::maxloc_decode<T>(buf->global.value), buf->global.rank };
    });
    return req;
}

template<typename T>
std::shared_ptr<MPIRequest> MPIComm::allreduce(const T* sendbuf, T* recvbuf,
                                               size_t count, MPI_Op op) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable for MPI communication");

    const size_t MAX_CHUNK_ELEMS = static_cast<size_t>(INT_MAX);
    // MPI forbids aliased send/recv buffers; in-place must be spelled
    // MPI_IN_PLACE (the input is then read from recvbuf).
    const bool in_place = recvbuf != nullptr && sendbuf == recvbuf;
    auto req = std::make_shared<MPIRequest>();
    size_t offset = 0;
    // do-while so count == 0 still issues one (legal) zero-count collective,
    // keeping every rank's collective sequence matched.
    do {
        const size_t chunk = std::min(MAX_CHUNK_ELEMS, count - offset);
        const void* src = in_place ? MPI_IN_PLACE
                                   : static_cast<const void*>(sendbuf + offset);
        STCOMM_MPI_CHECK(MPI_Iallreduce(src, recvbuf + offset,
                       static_cast<int>(chunk), detail::mpi_datatype<T>(), op,
                       comm_, &req->addHandle()));
        offset += chunk;
    } while (offset < count);
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
