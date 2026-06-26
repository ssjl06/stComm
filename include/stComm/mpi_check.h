#pragma once

#include <mpi.h>
#include <stdexcept>
#include <string>

namespace stComm {
namespace detail {

// Turn a failed MPI call into an exception. Note that MPI's default error
// handler (MPI_ERRORS_ARE_FATAL) aborts before returning, so this only fires
// once the caller installs MPI_ERRORS_RETURN — it is a defensive net, not the
// primary failure path.
[[noreturn]] inline void throwMpi(int code, const char* expr, const char* file,
                                  int line) {
    char buf[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(code, buf, &len);
    throw std::runtime_error(std::string("MPI error: ") +
                             std::string(buf, static_cast<size_t>(len)) +
                             " at " + file + ":" + std::to_string(line) +
                             " in '" + expr + "'");
}

}  // namespace detail
}  // namespace stComm

// Evaluate an MPI call and throw std::runtime_error on any non-success code.
#define STCOMM_MPI_CHECK(expr)                                                 \
    do {                                                                       \
        int _stcomm_mpi_code = (expr);                                         \
        if (_stcomm_mpi_code != MPI_SUCCESS)                                   \
            ::stComm::detail::throwMpi(_stcomm_mpi_code, #expr, __FILE__,      \
                                       __LINE__);                              \
    } while (0)
