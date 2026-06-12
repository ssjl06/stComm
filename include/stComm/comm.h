#pragma once

#include "mpi_comm.h"
#include "nccl_comm.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace stComm {

/**
 * @brief Memory space tag — selects the communication backend at compile time.
 *
 *   Host   → MPIComm  (host memory, MPI transport)
 *   Device → NCCLComm (device memory, NCCL transport)
 *
 * Used as the first template argument of Comm's space-selecting collectives.
 * Dispatch is via `if constexpr`, so only the chosen backend is instantiated
 * (a Host-only type never touches NCCLTypeMap) and there is no runtime branch.
 */
enum class Space { Host, Device };

namespace detail {

// Map the backend-agnostic ReduceOp to the concrete MPI_Op. Kept internal so
// callers of Comm never name MPI_Op themselves.
inline MPI_Op to_mpi_op(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum:  return MPI_SUM;
        case ReduceOp::Max:  return MPI_MAX;
        case ReduceOp::Min:  return MPI_MIN;
        case ReduceOp::Prod: return MPI_PROD;
    }
    return MPI_SUM;  // unreachable; silences -Wreturn-type
}

}  // namespace detail

/**
 * @brief Unified communication facade over MPI (host) and NCCL (device).
 *
 * One object owns an MPIComm and an optional NCCLComm and routes each collective
 * to the right backend by a compile-time Space tag. Because Comm is a concrete
 * class (not a polymorphic base) its operations stay templated — keeping full
 * compile-time type safety and zero-overhead dispatch while presenting a single
 * object to callers, instead of forcing them to juggle MPIComm + NCCLComm by hand.
 *
 * Both backends are held by unique_ptr for a uniform owning model: mpi_ is always
 * present, while nccl_ is null on a host-only Comm (see hasDevice()).
 *
 * Encapsulation: ordinary callers never name MPI_Comm / MPI_Op / ncclUniqueId /
 * CUDA streams. Backend specifics live on the backend objects, reachable by
 * advanced callers through the mpi() / nccl() escape hatches.
 *
 * Capability note: NCCL has no native MAXLOC or scan primitive, so the device
 * `allreduceMaxloc` / `exscan` are emulated (gather + host-side reduce); both
 * Spaces are async and return the corresponding request.
 */
class Comm {
public:
    /// @brief Request type returned by a space-selecting operation.
    ///
    /// Symmetric and concrete on both sides: Host yields `shared_ptr<MPIRequest>`,
    /// Device yields `shared_ptr<NCCLRequest>` — no cast needed to reach a native
    /// sync handle (MPIRequest::getHandle / NCCLRequest::getStream). A single
    /// MPIRequest tracks 1..N MPI_Request handles, so even >2GB chunked transfers
    /// stay one concrete type.
    template<Space Sp>
    using RequestPtrFor =
        std::conditional_t<Sp == Space::Device, NCCLRequestPtr, MPIRequestPtr>;

    // ---- Lifecycle (hides "MPI" from the init/finalize vocabulary) ----------
    static void initialize(int* argc, char*** argv) { MPIComm::initialize(argc, argv); }
    static void finalize()                          { MPIComm::finalize(); }

    // ---- Construction -------------------------------------------------------

    // Host-only communicator wrapping an existing MPI communicator.
    explicit Comm(MPI_Comm mpi = MPI_COMM_WORLD)
        : mpi_(std::make_unique<MPIComm>(mpi)) {}

    // Host + device communicator on `device_id`. Bootstraps NCCL internally:
    // rank 0 mints the ncclUniqueId and broadcasts it, then every rank calls
    // ncclCommInitRank (which also cudaSetDevice's `device_id`). **Collective** —
    // all ranks must call this together. Returns a prvalue, so C++17 guaranteed
    // copy elision constructs it in place despite Comm being non-movable.
    static Comm onDevice(int device_id, MPI_Comm mpi = MPI_COMM_WORLD) {
        return Comm(mpi, device_id);
    }

    // A Comm owns live MPI/NCCL state; non-copyable and non-movable, matching
    // MPIComm/NCCLComm. Hold it by reference or via a smart pointer.
    Comm(const Comm&)            = delete;
    Comm& operator=(const Comm&) = delete;

    int  getRank()   const { return mpi_->getRank(); }
    int  getSize()   const { return mpi_->getSize(); }
    bool hasDevice() const { return static_cast<bool>(nccl_); }
    void barrier()         { mpi_->barrier(); }

    // ---- Space-selecting collectives ------------------------------------
    //   Sp == Host   → host memory via MPI
    //   Sp == Device → device memory via NCCL (requires a device-enabled Comm)

    // Backends return their concrete request type already, so each branch
    // forwards directly — no cast, the RequestPtrFor<Sp> return type just lines
    // up with whichever backend handled it.
    template<Space Sp, typename T>
    RequestPtrFor<Sp> bcast(T* data, std::size_t count, int root) {
        if constexpr (Sp == Space::Device) { assert(nccl_); return nccl_->bcast(data, count, root); }
        else                                 return mpi_->bcast(data, count, root);
    }

    template<Space Sp, typename T>
    RequestPtrFor<Sp> allgatherv(const T* sendbuf, int sendcount,
                                 T* recvbuf, const int* recvcounts) {
        if constexpr (Sp == Space::Device) { assert(nccl_); return nccl_->allgatherv(sendbuf, sendcount, recvbuf, recvcounts); }
        else                                 return mpi_->allgatherv(sendbuf, sendcount, recvbuf, recvcounts);
    }

    template<Space Sp, typename T>
    RequestPtrFor<Sp> alltoallv(const T* sendbuf, const int* sendcounts,
                                T* recvbuf, const int* recvcounts) {
        if constexpr (Sp == Space::Device) { assert(nccl_); return nccl_->alltoallv(sendbuf, sendcounts, recvbuf, recvcounts); }
        else                                 return mpi_->alltoallv(sendbuf, sendcounts, recvbuf, recvcounts);
    }

    // ---- Space-selecting reductions (async) -----------------------------
    // Like the collectives above, these return a request; the scalar result is
    // delivered to *out once it completes. Host uses MPI_Iallreduce/Iexscan;
    // Device emulates via ncclAllGather + a host-side reduce (NCCL has no native
    // MAXLOC/scan). ReduceOp is mapped to the MPI op on the host path.
    template<Space Sp, typename T>
    RequestPtrFor<Sp> allreduceMaxloc(T value, std::pair<T, int>* out) {
        if constexpr (Sp == Space::Device) { assert(nccl_); return nccl_->allreduceMaxloc(value, out); }
        else                                 return mpi_->allreduceMaxloc(value, out);
    }

    template<Space Sp, typename T>
    RequestPtrFor<Sp> exscan(T value, T* out, ReduceOp op = ReduceOp::Sum) {
        if constexpr (Sp == Space::Device) { assert(nccl_); return nccl_->exscan(value, out, op); }
        else                                 return mpi_->exscan(value, out, detail::to_mpi_op(op));
    }

    // ---- Backend escape hatches for backend-specific APIs ---------------
    // Advanced callers reach MPI/NCCL specifics (native handles, streams,
    // groups) through the backend object itself; Comm stays backend-agnostic.
    MPIComm&  mpi()  { return *mpi_; }
    NCCLComm& nccl() { assert(nccl_); return *nccl_; }

private:
    // onDevice() only. Private so it can't be confused with the host ctor — note
    // MPICH types MPI_Comm as int, making Comm(MPI_Comm, int) ambiguous with a
    // hypothetical public two-arg overload.
    Comm(MPI_Comm mpi, int device_id)
        : mpi_(std::make_unique<MPIComm>(mpi)) {
        ncclUniqueId id;
        if (mpi_->getRank() == 0) id = NCCLComm::getUniqueId();
        mpi_->bcast<ncclUniqueId>(&id, 1, 0)->wait();
        nccl_ = std::make_unique<NCCLComm>();
        nccl_->initialize(mpi_->getRank(), mpi_->getSize(), device_id, id);
    }

    std::unique_ptr<MPIComm>  mpi_;    // always present
    std::unique_ptr<NCCLComm> nccl_;   // null ⇒ host-only
};

} // namespace stComm
