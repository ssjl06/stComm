#ifndef STCOMM_COMM_BASE_H
#define STCOMM_COMM_BASE_H

#include "types.h"

namespace stComm {

/**
 * @brief Backend type for communication
 */
enum class Backend {
    MPI,   ///< MPI backend for CPU communication
    NCCL   ///< NCCL backend for GPU communication
};

/**
 * @brief Abstract base class for all communicators
 *
 * Provides common interface for both MPI and NCCL backends.
 * Non-template methods are defined as virtual functions.
 */
class CommBase {
public:
    virtual ~CommBase() = default;

    /**
     * @brief Get the rank of this process
     * @return Rank ID (0 to size-1)
     */
    virtual int getRank() const = 0;

    /**
     * @brief Get the total number of processes
     * @return Number of processes
     */
    virtual int getSize() const = 0;

    /**
     * @brief Synchronization barrier
     *
     * Blocks until all processes in the communicator have called this function.
     */
    virtual void barrier() = 0;

    /**
     * @brief Get the backend type
     * @return Backend enum (MPI or NCCL)
     */
    virtual Backend getBackend() const = 0;

    /**
     * @brief Check if communicator is initialized
     * @return true if initialized, false otherwise
     */
    virtual bool isInitialized() const = 0;

protected:
    CommBase() = default;

    // Disable copy
    CommBase(const CommBase&) = delete;
    CommBase& operator=(const CommBase&) = delete;

    // Enable move
    CommBase(CommBase&&) = default;
    CommBase& operator=(CommBase&&) = default;
};

} // namespace stComm

#endif // STCOMM_COMM_BASE_H
