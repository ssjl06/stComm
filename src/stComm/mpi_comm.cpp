#include "stComm/mpi_comm.h"
#include <stdexcept>

namespace stComm {

MPIComm::MPIComm() : comm_(MPI_COMM_WORLD), rank_(0), size_(1) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

MPIComm::MPIComm(MPI_Comm comm) : comm_(comm), rank_(0), size_(1) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

void MPIComm::initialize(int* argc, char*** argv) {
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE) {
        throw std::runtime_error("MPI implementation does not support MPI_THREAD_MULTIPLE");
    }
}

void MPIComm::finalize() {
    MPI_Finalize();
}

int MPIComm::getRank() const {
    return rank_;
}

int MPIComm::getSize() const {
    return size_;
}

void MPIComm::barrier() {
    MPI_Barrier(comm_);
}

} // namespace stComm
