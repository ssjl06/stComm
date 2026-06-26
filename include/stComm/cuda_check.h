#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace stComm {
namespace detail {

// Turn a failed CUDA runtime call into an exception. Kept out of line of the
// macro so the hot (success) path stays a single comparison.
[[noreturn]] inline void throwCuda(cudaError_t err, const char* expr,
                                   const char* file, int line) {
    throw std::runtime_error(
        std::string("CUDA error: ") + cudaGetErrorString(err) + " (" +
        cudaGetErrorName(err) + ") at " + file + ":" + std::to_string(line) +
        " in '" + expr + "'");
}

}  // namespace detail
}  // namespace stComm

// Evaluate a CUDA runtime call and throw std::runtime_error on any non-success
// status. Use everywhere a cudaError_t is returned; never in a destructor.
#define STCOMM_CUDA_CHECK(expr)                                                \
    do {                                                                       \
        cudaError_t _stcomm_cuda_err = (expr);                                 \
        if (_stcomm_cuda_err != cudaSuccess)                                   \
            ::stComm::detail::throwCuda(_stcomm_cuda_err, #expr, __FILE__,     \
                                        __LINE__);                             \
    } while (0)
