#pragma once

#include <nccl.h>
#include <stdexcept>
#include <string>

namespace stComm {
namespace detail {

// Turn a failed NCCL call into an exception. Out of line of the macro so the
// success path stays a single comparison.
[[noreturn]] inline void throwNccl(ncclResult_t res, const char* expr,
                                   const char* file, int line) {
    throw std::runtime_error(
        std::string("NCCL error: ") + ncclGetErrorString(res) + " at " + file +
        ":" + std::to_string(line) + " in '" + expr + "'");
}

}  // namespace detail
}  // namespace stComm

// Evaluate an NCCL call and throw std::runtime_error on any non-success result.
// Never use in a destructor (it must not throw).
#define STCOMM_NCCL_CHECK(expr)                                                \
    do {                                                                       \
        ncclResult_t _stcomm_nccl_res = (expr);                                \
        if (_stcomm_nccl_res != ncclSuccess)                                   \
            ::stComm::detail::throwNccl(_stcomm_nccl_res, #expr, __FILE__,     \
                                        __LINE__);                             \
    } while (0)
