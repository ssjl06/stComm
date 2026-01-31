#ifndef STCOMM_UTILS_H
#define STCOMM_UTILS_H

#include <vector>
#include <numeric>
#include <cstring>

namespace stComm {

/**
 * @brief Utility functions for displacement calculation and data management
 */
class Utils {
public:
    /**
     * @brief Calculate displacements from counts
     * @param counts Array of counts for each rank
     * @param num_ranks Number of ranks
     * @return Vector of displacements (cumulative sum starting from 0)
     *
     * Example: counts = [10, 20, 30] -> displs = [0, 10, 30]
     */
    static std::vector<int> calculate_displacements(const int* counts, int num_ranks) {
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
    static std::vector<int> calculate_displacements(const std::vector<int>& counts) {
        return calculate_displacements(counts.data(), counts.size());
    }

    /**
     * @brief Calculate total size from counts
     */
    static int total_size(const int* counts, int num_ranks) {
        return std::accumulate(counts, counts + num_ranks, 0);
    }

    /**
     * @brief Calculate total size from counts (vector version)
     */
    static int total_size(const std::vector<int>& counts) {
        return std::accumulate(counts.begin(), counts.end(), 0);
    }

    /**
     * @brief Allocate buffer based on counts
     */
    template<typename T>
    static T* allocate_buffer(const int* counts, int num_ranks) {
        int size = total_size(counts, num_ranks);
        return new T[size];
    }

    /**
     * @brief Allocate buffer based on counts (vector version)
     */
    template<typename T>
    static T* allocate_buffer(const std::vector<int>& counts) {
        return allocate_buffer<T>(counts.data(), counts.size());
    }
};

/**
 * @brief RAII wrapper for auto-managed communication buffers
 */
template<typename T>
class CommBuffer {
public:
    CommBuffer() : data_(nullptr), size_(0) {}

    explicit CommBuffer(size_t size) : size_(size) {
        data_ = new T[size];
    }

    CommBuffer(const int* counts, int num_ranks) {
        size_ = Utils::total_size(counts, num_ranks);
        data_ = new T[size_];
    }

    CommBuffer(const std::vector<int>& counts) {
        size_ = Utils::total_size(counts);
        data_ = new T[size_];
    }

    ~CommBuffer() {
        if (data_) {
            delete[] data_;
        }
    }

    // Move semantics
    CommBuffer(CommBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    CommBuffer& operator=(CommBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Delete copy
    CommBuffer(const CommBuffer&) = delete;
    CommBuffer& operator=(const CommBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }

    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }

private:
    T* data_;
    size_t size_;
};

} // namespace stComm

#endif // STCOMM_UTILS_H
