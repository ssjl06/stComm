#pragma once

#include <cstdint>
#include <memory>

namespace stComm {

/**
 * @brief Backend type enumeration
 */
enum class Backend {
    MPI,   ///< MPI backend for CPU communication
    NCCL   ///< NCCL backend for GPU communication
};

/**
 * @brief Reduction operation, backend-agnostic.
 *
 * Lets callers request a reduction without naming MPI_Op / ncclRedOp_t. The
 * Comm facade maps it to the concrete backend op internally.
 */
enum class ReduceOp {
    Sum,
    Max,
    Min,
    Prod
};

/**
 * @brief Operation status
 */
enum class Status {
    SUCCESS = 0,
    PENDING,
    ERROR
};

// Forward declarations
class Request;
using RequestPtr = std::shared_ptr<Request>;

} // namespace stComm
