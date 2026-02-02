#include "stComm/mpi_comm.h"

namespace stComm {

MPIComm::MPIComm() : comm_(MPI_COMM_WORLD) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

MPIComm::MPIComm(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

void MPIComm::initialize(int* argc, char*** argv) {
    MPI_Init(argc, argv);
}

void MPIComm::finalize() {
    MPI_Finalize();
}

void MPIComm::barrier() {
    MPI_Barrier(comm_);
}

} // namespace stComm
