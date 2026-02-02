# stComm - High-Performance Communication Library

stComm is a C++/CUDA library providing template-based asynchronous communication primitives for distributed computing, supporting both MPI and NCCL backends.

## Features

- **Dual Backend Support**: MPI for CPU communication, NCCL for GPU communication
- **Object-Oriented Design**: Abstract base class with concrete implementations
- **Template-Based API**: Support for all primitive types (int, float, double, etc.)
- **Automatic Displacement Calculation**: No manual offset computation needed
- **Asynchronous Operations**: Non-blocking communication with async/await pattern
- **Large Data Support**: Uses MPI_BYTE to handle arbitrary data sizes efficiently
- **Type-Safe**: Compile-time type checking with templates

## Architecture

```
CommBase (Abstract Interface)
    ├── MPIComm (CPU/Host Memory)
    └── NCCLComm (GPU/Device Memory)
```

**Key Components:**
- `CommBase`: Abstract interface defining common operations
- `MPIComm`: MPI-based implementation for host memory
- `NCCLComm`: NCCL-based implementation for device memory
- `Request`: Async operation handle (supports both MPI_Request and CUDA streams)
- `Utils`: Automatic displacement calculation utilities

## Prerequisites

- GCC/G++ compiler with C++17 support
- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher
- OpenMPI or MPICH
- NCCL 2.0 or higher
- Google Test (for building tests)

## Quick Start

### 1. Configure Environment

```bash
cp env.sh.example env.sh
vim env.sh  # Edit paths for your system
```

Set the following in `env.sh`:
- `GCC_HOME`: GCC compiler installation path
- `CMAKE_HOME`: CMake installation path
- `CUDA_HOME`: CUDA Toolkit path
- `MPI_HOME`: OpenMPI/MPICH path
- `NCCL_HOME`: NCCL library path
- `GTEST_HOME`: Google Test path
- `CUDA_ARCH`: Your GPU architecture (75, 80, 89, etc.)

### 2. Build

```bash
source env.sh
./build.sh           # Release build
./build.sh --debug   # Debug build
./build.sh --clean   # Clean build
./build.sh -j 8      # Parallel build with 8 jobs
```

Outputs:
- `build/lib/libstComm.so`: Shared library
- `build/tests/stComm_tests`: Test binary

### 3. Run Tests

```bash
./run_tests.sh         # Run with 2 MPI processes
./run_tests.sh -np 4   # Run with 4 processes
./run_tests.sh -f "MPICommTest.*"  # Run specific tests
```

## Usage Examples

### MPI Communication (CPU)

```cpp
#include "stComm/stComm.h"
#include <vector>

int main(int argc, char** argv) {
    // Initialize MPI
    stComm::MPIComm::initialize(&argc, &argv);

    // Create communicator
    stComm::MPIComm comm;
    int rank = comm.getRank();
    int size = comm.getSize();

    // Point-to-point communication
    std::vector<double> data(1000);

    if (rank == 0) {
        // Fill data
        for (int i = 0; i < 1000; ++i) {
            data[i] = i * 1.5;
        }

        // Async send
        auto req = comm.send(data.data(), data.size(), 1, 0);

        // Do other work...

        // Wait for completion
        req->wait();
    } else if (rank == 1) {
        // Async receive
        auto req = comm.recv(data.data(), data.size(), 0, 0);
        req->wait();
    }

    // Collective communication with auto displacement
    std::vector<int> sendcounts(size, 10);
    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = i + 1;  // Variable counts
    }

    int my_count = rank + 1;
    std::vector<int> sendbuf(my_count, rank);

    int total_recv = stComm::Utils::totalSize(recvcounts);
    std::vector<int> recvbuf(total_recv);

    // Displacement is automatically calculated!
    auto req = comm.allgatherv(sendbuf.data(), my_count,
                               recvbuf.data(), recvcounts.data());
    req->wait();

    // Finalize
    stComm::MPIComm::finalize();
    return 0;
}
```

### NCCL Communication (GPU)

```cpp
#include "stComm/stComm.h"
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    // Initialize MPI for coordination
    stComm::MPIComm::initialize(&argc, &argv);
    stComm::MPIComm mpi_comm;

    int rank = mpi_comm.getRank();
    int size = mpi_comm.getSize();
    int device_id = rank % 4;

    // Get NCCL unique ID
    ncclUniqueId nccl_id;
    if (rank == 0) {
        nccl_id = stComm::NCCLComm::getUniqueId();
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    stComm::NCCLComm nccl_comm;
    nccl_comm.initialize(rank, size, device_id, nccl_id);

    // Allocate GPU memory
    const int N = 1000;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data on GPU
    std::vector<float> h_data(N, rank * 1.0f);
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Async send on GPU (uses internal CUDA stream)
    if (rank == 0) {
        auto req = nccl_comm.send(d_data, N, 1);
        req->wait();
    } else if (rank == 1) {
        auto req = nccl_comm.recv(d_data, N, 0);
        req->wait();
    }

    // Allgatherv with auto displacement
    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (i + 1) * 10;
    }

    int my_count = (rank + 1) * 10;
    float *d_sendbuf, *d_recvbuf;
    int total_recv = stComm::Utils::totalSize(recvcounts);
    cudaMalloc(&d_sendbuf, my_count * sizeof(float));
    cudaMalloc(&d_recvbuf, total_recv * sizeof(float));

    // Displacement auto-calculated! Uses internal stream for async operations
    auto req = nccl_comm.allgatherv(d_sendbuf, my_count,
                                    d_recvbuf, recvcounts.data());
    req->wait();

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);

    stComm::MPIComm::finalize();
    return 0;
}
```

## API Reference

### MPIComm

**Static Methods:**
- `initialize(int* argc, char*** argv)`: Initialize MPI
- `finalize()`: Finalize MPI

**Instance Methods:**
- `int getRank()`: Get process rank
- `int getSize()`: Get total processes
- `void barrier()`: Synchronization barrier
- `send<T>(data, count, dest, tag)`: Async send
- `recv<T>(data, count, source, tag)`: Async receive
- `allgatherv<T>(sendbuf, sendcount, recvbuf, recvcounts)`: Variable-length allgather (auto displacement)
- `alltoallv<T>(sendbuf, sendcounts, recvbuf, recvcounts)`: Variable-length alltoall (auto displacement)

### NCCLComm

**Static Methods:**
- `getUniqueId()`: Get NCCL unique ID (call on rank 0)

**Instance Methods:**
- `initialize(rank, nranks, device_id, comm_id)`: Initialize NCCL (creates internal CUDA stream)
- `int getRank()`: Get process rank
- `int getSize()`: Get total processes
- `send<T>(data, count, dest)`: Async send on GPU (uses internal stream)
- `recv<T>(data, count, source)`: Async receive on GPU (uses internal stream)
- `allgatherv<T>(sendbuf, sendcount, recvbuf, recvcounts)`: Variable-length allgather (auto displacement, uses internal stream)
- `alltoallv<T>(sendbuf, sendcounts, recvbuf, recvcounts)`: Variable-length alltoall (auto displacement, uses internal stream)

**Note:** NCCLComm manages an internal CUDA stream created during initialization. All operations use this stream automatically for async execution.

### Utils

- `calculateDisplacements(counts, num_ranks)`: Auto calculate offsets from counts
- `totalSize(counts, num_ranks)`: Calculate total buffer size

### Request

- `wait()`: Wait for operation to complete
- `test()`: Test if operation is complete (non-blocking)
- `getStatus()`: Get operation status

## Key Features Explained

### Automatic Displacement Calculation

No need to manually calculate offsets:

```cpp
// OLD WAY (Manual)
int displs[size];
displs[0] = 0;
for (int i = 1; i < size; ++i) {
    displs[i] = displs[i-1] + recvcounts[i-1];
}

// NEW WAY (Automatic)
auto req = comm.allgatherv(sendbuf, sendcount, recvbuf, recvcounts);
// Displacement calculated internally!
```

### Template-Based Type Support

Works with any primitive type:

```cpp
comm.send<int>(int_data, count, dest);
comm.send<float>(float_data, count, dest);
comm.send<double>(double_data, count, dest);
```

### Asynchronous Operations

Non-blocking with explicit wait:

```cpp
auto req = comm.send(data, count, dest);
// Do other work...
req->wait();  // Wait when needed
```

## Supported Data Types

All trivially copyable types:
- `char`, `int8_t`, `int16_t`, `int32_t`, `int64_t`
- `unsigned char`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`
- `float`, `double`
- Any other trivially copyable struct

## License

This project is provided as-is for educational and research purposes.

## Authors

Samsung Research

## References

- [MPI Standard](https://www.mpi-forum.org/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
