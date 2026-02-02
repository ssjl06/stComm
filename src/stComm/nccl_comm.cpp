#include "stComm/nccl_comm.h"
#include <stdexcept>

namespace stComm {

NCCLComm::NCCLComm()
    : comm_(nullptr), rank_(-1), size_(0), device_id_(-1),
      initialized_(false), default_stream_(nullptr) {
}

NCCLComm::~NCCLComm() {
    if (initialized_ && comm_ != nullptr) {
        ncclCommDestroy(comm_);
    }
    if (default_stream_ != nullptr) {
        cudaStreamDestroy(default_stream_);
    }
}

void NCCLComm::initialize(int rank, int nranks, int device_id, ncclUniqueId comm_id) {
    rank_ = rank;
    size_ = nranks;
    device_id_ = device_id;

    // Set CUDA device
    cudaSetDevice(device_id_);

    // Create default stream
    cudaStreamCreate(&default_stream_);

    // Initialize NCCL communicator
    ncclCommInitRank(&comm_, nranks, comm_id, rank);

    initialized_ = true;
}

ncclUniqueId NCCLComm::getUniqueId() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    return id;
}

void NCCLComm::barrier() {
    // NCCL doesn't have native barrier, use stream synchronization
    if (default_stream_ != nullptr) {
        cudaStreamSynchronize(default_stream_);
    }
}

cudaStream_t NCCLComm::getOrCreateStream(cudaStream_t stream) {
    return stream != nullptr ? stream : default_stream_;
}

} // namespace stComm
