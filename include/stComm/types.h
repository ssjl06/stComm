#ifndef STCOMM_TYPES_H
#define STCOMM_TYPES_H

#include <cstdint>
#include <cstddef>

namespace stComm {

// Communication backend types
enum class Backend {
    MPI,
    NCCL
};

// Communication status
enum class Status {
    SUCCESS,
    PENDING,
    ERROR
};

// Data location (host or device)
enum class MemoryType {
    HOST,
    DEVICE
};

} // namespace stComm

#endif // STCOMM_TYPES_H
