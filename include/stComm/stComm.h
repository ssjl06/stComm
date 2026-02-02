#pragma once

/**
 * @file stComm.h
 * @brief Main header for stComm library
 *
 * Include this file to access all stComm functionality.
 */

#include "stComm/types.h"
#include "stComm/utils.h"
#include "stComm/request.h"
#include "stComm/comm_base.h"
#include "stComm/mpi_comm.h"
#include "stComm/nccl_comm.h"

namespace stComm {

// Library version
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

inline const char* getVersion() {
    return "1.0.0";
}

} // namespace stComm
