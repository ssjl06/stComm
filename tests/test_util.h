#pragma once

#include <cstddef>

namespace stComm_test {

// Tests count ranks and elements in int (MPI's count type) but size and index
// std::vector with size_t. uz() is the one explicit int -> size_t conversion
// point, so -Wsign-conversion stays clean without a static_cast per subscript.
constexpr std::size_t uz(int n) { return static_cast<std::size_t>(n); }

} // namespace stComm_test
