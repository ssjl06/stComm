# stComm - High-Performance Communication Library

stComm is a C++/CUDA library providing unified asynchronous communication primitives for distributed computing, supporting both MPI and NCCL backends.

## Features

- **Unified Interface**: Single `Communicator` class works with both MPI and NCCL backends
- **Dual Backend Support**: Seamlessly use MPI for CPU-based communication or NCCL for GPU-based communication
- **High-Level APIs**: Auto displacement calculation and vector-based interfaces
- **Template-Based API**: Support for various primitive data types (int, float, double, etc.)
- **Asynchronous Operations**: Non-blocking communication with async/await pattern
- **Collective Operations**: Efficient implementations of allgatherv, alltoallv, and more
- **Large Data Support**: Handle data transfers efficiently regardless of size
- **Type-Safe**: Template-based design ensures compile-time type checking

## Architecture

```
stComm/
├── include/stComm/        # Public headers
│   ├── stComm.h          # Main header (include this)
│   ├── types.h           # Common type definitions
│   ├── request.h         # Async request handles
│   ├── comm_base.h       # Abstract base class
│   ├── communicator.h    # Unified interface (recommended)
│   ├── mpi_comm.h        # MPI communication interface
│   ├── nccl_comm.h       # NCCL communication interface
│   └── utils.h           # Utility functions (displacement calc, etc.)
├── src/stComm/           # Implementation
├── tests/                # Google Test suite
├── examples/             # Example programs
└── build/                # Build artifacts (generated)
    ├── lib/              # Shared library
    ├── tests/            # Test executables
    └── examples/         # Example binaries
```

## Prerequisites

- GCC/G++ compiler with C++17 support
- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher
- OpenMPI or MPICH
- NCCL 2.0 or higher
- Google Test (for building tests)

## Quick Start

### 1. Configure Environment

Copy the example environment configuration and edit paths:

```bash
cp env.sh.example env.sh
# Edit env.sh to set correct paths for your system
vim env.sh
```

Example env.sh configuration:
```bash
export GCC_HOME="/usr/local/gcc-11"
export CUDA_HOME="/usr/local/cuda-12.0"
export MPI_HOME="/usr/local/openmpi-4.1"
export NCCL_HOME="/usr/local/nccl-2.18"
export GTEST_HOME="/usr/local/gtest"
export CUDA_ARCH="89"  # Set to your GPU architecture
```

### 2. Build

```bash
# Release build
./build.sh

# Debug build
./build.sh --debug

# Clean build
./build.sh --clean

# Parallel build with 8 jobs
./build.sh -j 8
```

### 3. Run Tests

```bash
# Run all tests with 2 MPI processes
./run_tests.sh

# Run with 4 MPI processes
./run_tests.sh -np 4

# Run specific test suite
./run_tests.sh -f "MPICommTest.*"
```

## Usage Examples

### Unified Interface (Recommended)

The `Communicator` class provides a unified interface that works with both MPI and NCCL:

```cpp
#include "stComm/stComm.h"
#include <vector>

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);

    // Create communicator with MPI backend
    stComm::Communicator comm(stComm::Backend::MPI);
    // For NCCL: stComm::Communicator comm(stComm::Backend::NCCL, rank, size, device_id, nccl_id);

    int rank = comm.getRank();
    int size = comm.getSize();

    // High-level vector API - easiest!
    std::vector<int> my_data(10, rank);
    auto result = comm.allgather_vec(my_data);

    // Auto displacement - no manual calculation!
    std::vector<int> recvcounts(size);
    for (int i = 0; i < size; ++i) recvcounts[i] = i + 1;

    std::vector<int> sendbuf(rank + 1, rank * 100);
    std::vector<int> recvbuf(stComm::Utils::total_size(recvcounts));

    auto req = comm.allgatherv_auto(sendbuf.data(), sendbuf.size(),
                                    recvbuf.data(), recvcounts.data());
    req->wait();

    stComm::MPIComm::finalize();
    return 0;
}
```

**Switching backends is easy:**
1. Change `Backend::MPI` to `Backend::NCCL`
2. Use device memory instead of host memory
3. Everything else stays the same!

### MPI Communication (Direct)

```cpp
#include "stComm/stComm.h"
#include <vector>

int main(int argc, char** argv) {
    // Initialize MPI
    stComm::MPIComm::initialize(&argc, &argv);

    stComm::MPIComm comm;
    int rank = comm.getRank();
    int size = comm.getSize();

    // Point-to-point communication
    std::vector<double> data(1000);

    if (rank == 0) {
        // Non-blocking send
        auto req = comm.send(data.data(), data.size(), 1, 0);

        // Do other work here...

        // Wait for completion
        req->wait();
    } else if (rank == 1) {
        auto req = comm.recv(data.data(), data.size(), 0, 0);
        req->wait();
    }

    // Collective communication
    std::vector<int> sendcounts(size, 100);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        displs[i] = i * 100;
    }

    std::vector<int> sendbuf(100 + rank);
    std::vector<int> recvbuf(100 * size);

    auto req = comm.allgatherv(sendbuf.data(), sendbuf.size(),
                               recvbuf.data(), sendcounts.data(),
                               displs.data());
    req->wait();

    stComm::MPIComm::finalize();
    return 0;
}
```

### NCCL Communication

```cpp
#include "stComm/stComm.h"
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    // Initialize MPI for coordination
    stComm::MPIComm::initialize(&argc, &argv);
    stComm::MPIComm mpi_comm;

    int rank = mpi_comm.getRank();
    int size = mpi_comm.getSize();
    int device_id = rank % 4;  // Assuming 4 GPUs per node

    // Get NCCL unique ID from rank 0
    ncclUniqueId nccl_id;
    if (rank == 0) {
        nccl_id = stComm::NCCLComm::getUniqueId();
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    stComm::NCCLComm nccl_comm;
    nccl_comm.initialize(rank, size, device_id, nccl_id);

    // Allocate device memory
    float *dev_data;
    cudaMalloc(&dev_data, 1000 * sizeof(float));

    // AllReduce on GPU
    float *dev_result;
    cudaMalloc(&dev_result, 1000 * sizeof(float));

    auto req = nccl_comm.allreduce(dev_data, dev_result, 1000, ncclSum);
    req->wait();

    cudaFree(dev_data);
    cudaFree(dev_result);

    stComm::MPIComm::finalize();
    return 0;
}
```

## API Reference

### MPIComm

- `send<T>(data, count, dest, tag)` - Non-blocking send
- `recv<T>(data, count, source, tag)` - Non-blocking receive
- `allgatherv<T>(sendbuf, sendcount, recvbuf, recvcounts, displs)` - Variable-size allgather
- `alltoallv<T>(sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls)` - Variable-size alltoall
- `barrier()` - Synchronization barrier

### NCCLComm

- `send<T>(data, count, dest, stream)` - Non-blocking GPU send
- `recv<T>(data, count, source, stream)` - Non-blocking GPU receive
- `allgather<T>(sendbuf, recvbuf, count, stream)` - GPU allgather
- `allreduce<T>(sendbuf, recvbuf, count, op, stream)` - GPU allreduce

### Request

- `wait()` - Block until operation completes
- `test(completed)` - Check if operation is complete (non-blocking)

## Performance Considerations

1. **MPI Large Data**: The library automatically uses `MPI_BYTE` for primitive types to avoid MPI datatype limitations
2. **NCCL Streams**: Pass custom CUDA streams to overlap communication with computation
3. **Async Operations**: Use non-blocking calls and wait only when data is needed
4. **Memory Layout**: Ensure contiguous memory for best performance

## Building with stComm

### CMake Integration

```cmake
find_package(stComm REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp stComm::stComm)
```

### Manual Compilation

```bash
mpicxx -o myapp myapp.cpp \
    -I/path/to/stComm/include \
    -L/path/to/stComm/build/lib \
    -lstComm \
    -lcudart -lnccl
```

## Development

### Project Structure

- **include/stComm/**: Public API headers
- **src/stComm/**: Implementation files
- **tests/**: Unit and integration tests
- **cmake/**: CMake configuration files

### Adding New Features

1. Add header declarations in `include/stComm/`
2. Implement in `src/stComm/`
3. Add tests in `tests/`
4. Update CMakeLists.txt if adding new files
5. Rebuild and run tests

## Troubleshooting

### Build Issues

**Problem**: CMake cannot find NCCL
```bash
# Solution: Set NCCL_HOME in env.sh
export NCCL_HOME="/path/to/nccl"
```

**Problem**: CUDA architecture mismatch
```bash
# Solution: Set correct CUDA_ARCH in env.sh
export CUDA_ARCH="89"  # For Ada Lovelace GPUs
```

### Runtime Issues

**Problem**: Library not found at runtime
```bash
# Solution: Add library path to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/stComm/build/lib:$LD_LIBRARY_PATH
```

**Problem**: NCCL initialization fails
```bash
# Solution: Ensure all ranks use consistent NCCL unique ID
# Check network connectivity between nodes
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Authors

[Add authors/maintainers here]

## Version

Current version: 1.0.0

## Support

For issues and questions, please open an issue on the project repository.
