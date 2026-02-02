#ifndef STCOMM_UTILS_H
#define STCOMM_UTILS_H

#include <vector>
#include <numeric>

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
        std::vector<int> displs(num_ranks);
        displs[0] = 0;
        for (int i = 1; i < num_ranks; ++i) {
            displs[i] = displs[i - 1] + counts[i - 1];
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

#endif // STCOMM_UTILS_H
