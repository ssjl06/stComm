#pragma once

#include <climits>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace stComm {

/**
 * @brief Utility functions for communication operations
 */
class Utils {
public:
    /**
     * @brief Calculate displacements from counts
     *
     * Automatically computes cumulative displacements from an array of counts.
     * Example: counts=[10, 20, 30] → displs=[0, 10, 30]
     *
     * @param counts Array of counts for each rank
     * @param num_ranks Number of ranks
     * @return Vector of displacements
     */
    static std::vector<int> calculateDisplacements(const int* counts, int num_ranks) {
        // Accumulate in int64: an int running sum past INT_MAX is UB and
        // would hand MPI/NCCL a wrapped (negative) offset.
        std::vector<int> displs(num_ranks);
        std::int64_t offset = 0;
        for (int i = 0; i < num_ranks; ++i) {
            if (offset > INT_MAX) {
                throw std::overflow_error(
                    "stComm: displacement " + std::to_string(offset) +
                    " exceeds INT_MAX (counts sum past the int range)");
            }
            displs[i] = static_cast<int>(offset);
            offset += counts[i];
        }
        return displs;
    }

    /**
     * @brief Calculate displacements from counts (vector version)
     */
    static std::vector<int> calculateDisplacements(const std::vector<int>& counts) {
        return calculateDisplacements(counts.data(), counts.size());
    }

    /**
     * @brief Calculate total size from counts
     */
    static int totalSize(const int* counts, int num_ranks) {
        return std::accumulate(counts, counts + num_ranks, 0);
    }

    /**
     * @brief Calculate total size from counts (vector version)
     */
    static int totalSize(const std::vector<int>& counts) {
        return std::accumulate(counts.begin(), counts.end(), 0);
    }
};

} // namespace stComm
