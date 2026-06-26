#include "stComm/nccl_comm.h"
#include <stdexcept>

namespace stComm {

NCCLComm::NCCLComm()
    : comm_(nullptr), rank_(-1), size_(0), device_id_(-1),
      initialized_(false), stream_(nullptr) {
}

NCCLComm::~NCCLComm() {
    if (initialized_ && comm_ != nullptr) {
        ncclCommDestroy(comm_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void NCCLComm::initialize(int rank, int nranks, int device_id, ncclUniqueId comm_id) {
    rank_ = rank;
    size_ = nranks;
    device_id_ = device_id;

    // Set CUDA device
    STCOMM_CUDA_CHECK(cudaSetDevice(device_id_));

    // Create internal CUDA stream for all operations
    STCOMM_CUDA_CHECK(cudaStreamCreate(&stream_));

    // Initialize NCCL communicator
    STCOMM_NCCL_CHECK(ncclCommInitRank(&comm_, nranks, comm_id, rank));

    initialized_ = true;
}

ncclUniqueId NCCLComm::getUniqueId() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    return id;
}

void NCCLComm::barrier() {
    // NCCL doesn't have native barrier, use stream synchronization
    if (stream_ != nullptr) {
        STCOMM_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
}

void NCCLComm::groupStart() {
    STCOMM_NCCL_CHECK(ncclGroupStart());
    in_group_ = true;
}

void NCCLComm::groupEnd() {
    STCOMM_NCCL_CHECK(ncclGroupEnd());
    in_group_ = false;
    // The grouped ops are only now on the stream; record every parked request's
    // event at this single completion point (they all complete together).
    for (auto& req : pending_records_) {
        req->record(stream_);
    }
    pending_records_.clear();
}

void NCCLComm::recordOrDefer(const std::shared_ptr<NCCLRequest>& req) {
    if (in_group_) {
        pending_records_.push_back(req);
    } else {
        req->record(stream_);
    }
}

} // namespace stComm
