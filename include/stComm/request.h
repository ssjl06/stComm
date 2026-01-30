#ifndef STCOMM_REQUEST_H
#define STCOMM_REQUEST_H

#include "types.h"
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>

namespace stComm {

// Base class for async communication requests
class Request {
public:
    virtual ~Request() = default;

    // Wait for the operation to complete
    virtual Status wait() = 0;

    // Test if the operation has completed (non-blocking)
    virtual Status test(bool& completed) = 0;

    // Get the backend type
    virtual Backend getBackend() const = 0;
};

// MPI-based request
class MPIRequest : public Request {
public:
    MPIRequest();
    explicit MPIRequest(MPI_Request req);
    ~MPIRequest() override = default;

    Status wait() override;
    Status test(bool& completed) override;
    Backend getBackend() const override { return Backend::MPI; }

    MPI_Request& getHandle() { return request_; }

private:
    MPI_Request request_;
};

// NCCL-based request (uses CUDA streams)
class NCCLRequest : public Request {
public:
    NCCLRequest();
    explicit NCCLRequest(cudaStream_t stream);
    ~NCCLRequest() override;

    Status wait() override;
    Status test(bool& completed) override;
    Backend getBackend() const override { return Backend::NCCL; }

    cudaStream_t getStream() const { return stream_; }

private:
    cudaStream_t stream_;
    bool owns_stream_;
};

using RequestPtr = std::shared_ptr<Request>;

} // namespace stComm

#endif // STCOMM_REQUEST_H
