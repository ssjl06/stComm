#include "stComm/communicator.h"
#include <stdexcept>

namespace stComm {

// Constructor for MPI backend
Communicator::Communicator(Backend backend)
    : backend_(backend), mpi_comm_(nullptr), nccl_comm_(nullptr) {

    if (backend != Backend::MPI) {
        throw std::invalid_argument("Use the other constructor for NCCL backend");
    }

    mpi_comm_ = std::make_unique<MPIComm>();
}

// Constructor for NCCL backend
Communicator::Communicator(Backend backend, int rank, int nranks, int device_id, ncclUniqueId comm_id)
    : backend_(backend), mpi_comm_(nullptr), nccl_comm_(nullptr) {

    if (backend != Backend::NCCL) {
        throw std::invalid_argument("Use the other constructor for MPI backend");
    }

    nccl_comm_ = std::make_unique<NCCLComm>();
    nccl_comm_->initialize(rank, nranks, device_id, comm_id);
}

int Communicator::getRank() const {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->getRank();
    } else {
        return nccl_comm_->getRank();
    }
}

int Communicator::getSize() const {
    if (backend_ == Backend::MPI) {
        return mpi_comm_->getSize();
    } else {
        return nccl_comm_->getSize();
    }
}

void Communicator::barrier() {
    if (backend_ == Backend::MPI) {
        mpi_comm_->barrier();
    } else {
        // NCCL doesn't have barrier, could use synchronization or throw
        throw std::runtime_error("NCCL backend does not support barrier");
    }
}

bool Communicator::isInitialized() const {
    if (backend_ == Backend::MPI) {
        return mpi_comm_ != nullptr;
    } else {
        return nccl_comm_ != nullptr;
    }
}

} // namespace stComm
