#include "stComm/request.h"
#include <stdexcept>

namespace stComm {

// MPIRequest implementation
MPIRequest::MPIRequest() : request_(MPI_REQUEST_NULL) {}

MPIRequest::MPIRequest(MPI_Request req) : request_(req) {}

Status MPIRequest::wait() {
    if (request_ == MPI_REQUEST_NULL) {
        return Status::SUCCESS;
    }

    MPI_Status status;
    int result = MPI_Wait(&request_, &status);

    if (result == MPI_SUCCESS) {
        return Status::SUCCESS;
    } else {
        return Status::ERROR;
    }
}

Status MPIRequest::test(bool& completed) {
    if (request_ == MPI_REQUEST_NULL) {
        completed = true;
        return Status::SUCCESS;
    }

    int flag = 0;
    MPI_Status status;
    int result = MPI_Test(&request_, &flag, &status);

    completed = (flag != 0);

    if (result == MPI_SUCCESS) {
        return completed ? Status::SUCCESS : Status::PENDING;
    } else {
        return Status::ERROR;
    }
}

// NCCLRequest implementation
NCCLRequest::NCCLRequest() : stream_(nullptr), owns_stream_(false) {}

NCCLRequest::NCCLRequest(cudaStream_t stream)
    : stream_(stream), owns_stream_(stream == nullptr) {
    if (owns_stream_) {
        cudaStreamCreate(&stream_);
    }
}

NCCLRequest::~NCCLRequest() {
    if (owns_stream_ && stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

Status NCCLRequest::wait() {
    if (stream_ == nullptr) {
        return Status::SUCCESS;
    }

    cudaError_t result = cudaStreamSynchronize(stream_);

    if (result == cudaSuccess) {
        return Status::SUCCESS;
    } else {
        return Status::ERROR;
    }
}

Status NCCLRequest::test(bool& completed) {
    if (stream_ == nullptr) {
        completed = true;
        return Status::SUCCESS;
    }

    cudaError_t result = cudaStreamQuery(stream_);

    if (result == cudaSuccess) {
        completed = true;
        return Status::SUCCESS;
    } else if (result == cudaErrorNotReady) {
        completed = false;
        return Status::PENDING;
    } else {
        completed = false;
        return Status::ERROR;
    }
}

} // namespace stComm
