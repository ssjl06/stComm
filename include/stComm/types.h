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
