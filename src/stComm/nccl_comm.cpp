#include "stComm/nccl_comm.h"
#include <stdexcept>
#include <sstream>

namespace stComm {

NCCLComm::NCCLComm()
    : comm_(nullptr), rank_(0), size_(1), device_id_(0), initialized_(false) {}

NCCLComm::~NCCLComm() {
    if (initialized_ && comm_ != nullptr) {
        ncclCommDestroy(comm_);
    }
}

void NCCLComm::initialize(int rank, int nranks, int device_id, ncclUniqueId comm_id) {
    rank_ = rank;
    size_ = nranks;
    device_id_ = device_id;

    // Set CUDA device
    cudaError_t cuda_result = cudaSetDevice(device_id_);
    if (cuda_result != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to set CUDA device " << device_id_ << ": "
            << cudaGetErrorString(cuda_result);
        throw std::runtime_error(oss.str());
    }

    // Initialize NCCL communicator
    ncclResult_t nccl_result = ncclCommInitRank(&comm_, nranks, comm_id, rank);
    if (nccl_result != ncclSuccess) {
        std::ostringstream oss;
        oss << "Failed to initialize NCCL communicator: "
            << ncclGetErrorString(nccl_result);
        throw std::runtime_error(oss.str());
    }

    initialized_ = true;
}

ncclUniqueId NCCLComm::getUniqueId() {
    ncclUniqueId id;
    ncclResult_t result = ncclGetUniqueId(&id);

    if (result != ncclSuccess) {
        std::ostringstream oss;
        oss << "Failed to get NCCL unique ID: " << ncclGetErrorString(result);
        throw std::runtime_error(oss.str());
    }

    return id;
}

cudaStream_t NCCLComm::getOrCreateStream(cudaStream_t stream) {
    if (stream != nullptr) {
        return stream;
    }

    // Create a new stream for this operation
    cudaStream_t new_stream;
    cudaError_t result = cudaStreamCreate(&new_stream);

    if (result != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to create CUDA stream: " << cudaGetErrorString(result);
        throw std::runtime_error(oss.str());
    }

    return new_stream;
}

} // namespace stComm
