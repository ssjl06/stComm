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
 * Capability note: NCCL has no MAXLOC or scan primitive, so `allreduceMaxloc`
 * and `exscan` currently run on the MPI (host) backend only. Device variants are
 * a planned follow-up — adding them will give those two methods a Space tag too.
 */
class Comm {
public:
    /// @brief Request type returned by a space-selecting collective.
    ///
    /// Device collectives always produce an NCCLRequest, so the return is the
    /// concrete `shared_ptr<NCCLRequest>` — no cast needed to reach getStream().
    /// Host collectives may return MPIRequest or (for >2GB chunked transfers)
    /// MultiMPIRequest, so the return stays the base RequestPtr; identify the
    /// concrete type via Request::getBackend() if a native handle is needed.
    template<Space Sp>
    using RequestPtrFor =
        std::conditional_t<Sp == Space::Device, std::shared_ptr<NCCLRequest>, RequestPtr>;

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

    template<Space Sp, typename T>
    RequestPtrFor<Sp> bcast(T* data, std::size_t count, int root) {
        if constexpr (Sp == Space::Device) {
            assert(nccl_);
            return std::static_pointer_cast<NCCLRequest>(nccl_->bcast(data, count, root));
        } else {
            return mpi_->bcast(data, count, root);
        }
    }

    template<Space Sp, typename T>
    RequestPtrFor<Sp> allgatherv(const T* sendbuf, int sendcount,
                                 T* recvbuf, const int* recvcounts) {
        if constexpr (Sp == Space::Device) {
            assert(nccl_);
            return std::static_pointer_cast<NCCLRequest>(
                nccl_->allgatherv(sendbuf, sendcount, recvbuf, recvcounts));
        } else {
            return mpi_->allgatherv(sendbuf, sendcount, recvbuf, recvcounts);
        }
    }

    template<Space Sp, typename T>
    RequestPtrFor<Sp> alltoallv(const T* sendbuf, const int* sendcounts,
                                T* recvbuf, const int* recvcounts) {
        if constexpr (Sp == Space::Device) {
            assert(nccl_);
            return std::static_pointer_cast<NCCLRequest>(
                nccl_->alltoallv(sendbuf, sendcounts, recvbuf, recvcounts));
        } else {
            return mpi_->alltoallv(sendbuf, sendcounts, recvbuf, recvcounts);
        }
    }

    // ---- Host control-plane ops (no NCCL primitive) ---------------------
    // Device variants are a planned follow-up (will add a Space tag).
    template<typename T>
    std::pair<T, int> allreduceMaxloc(T value) { return mpi_->allreduceMaxloc(value); }

    template<typename T>
    T exscan(T value, ReduceOp op = ReduceOp::Sum) {
        return mpi_->exscan(value, detail::to_mpi_op(op));
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
