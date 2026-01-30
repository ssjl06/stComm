#ifndef STCOMM_H
#define STCOMM_H

// Main header file for stComm library
// Includes all necessary components for MPI and NCCL communication

#include "stComm/types.h"
#include "stComm/request.h"
#include "stComm/mpi_comm.h"
#include "stComm/nccl_comm.h"

namespace stComm {

// Library version
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// Utility function to get version string
const char* getVersion();

} // namespace stComm

#endif // STCOMM_H
