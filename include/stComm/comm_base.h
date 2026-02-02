#pragma once

#include "types.h"
#include "request.h"
#include <cuda_runtime.h>

namespace stComm {

/**
 * @brief Abstract base class for communication backends
 *
 * Defines common interface for both MPI and NCCL communication.
 * All communication operations are template-based to support various primitive types.
 */
class CommBase {
public:
    virtual ~CommBase() = default;

    /**
     * @brief Get process rank
     * @return Rank ID (0 to size-1)
     */
    virtual int getRank() const = 0;

    /**
     * @brief Get total number of processes
     * @return Number of processes
     */
    virtual int getSize() const = 0;

    /**
     * @brief Get backend type
     * @return Backend enum (MPI or NCCL)
     */
    virtual Backend getBackend() const = 0;

    /**
     * @brief Synchronization barrier
     *
     * Blocks until all processes reach this point.
     * Note: NCCL doesn't natively support barrier.
     */
    virtual void barrier() = 0;

    // Point-to-point communication is template-based and defined in derived classes
    // Collective communication is template-based and defined in derived classes

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
