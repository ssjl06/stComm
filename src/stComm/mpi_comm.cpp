#include "stComm/mpi_comm.h"

namespace stComm {

MPIComm::MPIComm() : comm_(MPI_COMM_WORLD) {
    STCOMM_MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
    STCOMM_MPI_CHECK(MPI_Comm_size(comm_, &size_));
}

MPIComm::MPIComm(MPI_Comm comm) : comm_(comm) {
    STCOMM_MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
    STCOMM_MPI_CHECK(MPI_Comm_size(comm_, &size_));
}

void MPIComm::initialize(int* argc, char*** argv) {
    STCOMM_MPI_CHECK(MPI_Init(argc, argv));
}

void MPIComm::finalize() {
    STCOMM_MPI_CHECK(MPI_Finalize());
}

void MPIComm::barrier() {
    STCOMM_MPI_CHECK(MPI_Barrier(comm_));
}

} // namespace stComm
