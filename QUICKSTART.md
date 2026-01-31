# stComm Quick Start Guide

## What Makes stComm Easy to Use?

stComm provides **high-level APIs** that eliminate manual work:

- ✅ **Auto displacement calculation** - No more manual offset math
- ✅ **Vector-based APIs** - Return results directly, no buffer management
- ✅ **Template-based** - Works with any primitive type
- ✅ **Async by default** - Non-blocking operations for better performance

### Quick Comparison

**Traditional MPI (allgatherv with variable lengths):**
```cpp
// Manual displacement calculation - tedious!
int displs[size];
displs[0] = 0;
for (int i = 1; i < size; ++i) {
    displs[i] = displs[i-1] + recvcounts[i-1];
}
MPI_Iallgatherv(sendbuf, sendcount, MPI_BYTE,
                recvbuf, recvcounts, displs, MPI_BYTE,
                comm, &req);
```

**stComm (auto displacement):**
```cpp
// Displacement calculated automatically!
auto req = comm.allgatherv_auto(sendbuf, sendcount, recvbuf, recvcounts);
```

**stComm (vector API - even simpler):**
```cpp
// One line! Returns result directly
auto result = comm.allgatherv_vec(sendbuf, recvcounts);
```

## 1. Setup Environment (First Time Only)

Edit `env.sh` to configure paths for your system:

```bash
vim env.sh
```

Set the following paths:
- `GCC_HOME`: GCC compiler installation path
- `CMAKE_HOME`: CMake installation path
- `CUDA_HOME`: CUDA Toolkit installation path
- `MPI_HOME`: OpenMPI/MPICH installation path
- `NCCL_HOME`: NCCL library installation path
- `GTEST_HOME`: Google Test installation path
- `CUDA_ARCH`: Your GPU architecture (e.g., "75" for Turing, "80" for Ampere, "89" for Ada)

## 2. Build the Library

```bash
# Release build (optimized)
./build.sh

# Or debug build (with debug symbols)
./build.sh --debug

# Clean build
./build.sh --clean
```

Build outputs:
- Shared library: `build/lib/libstComm.so`
- Test executable: `build/tests/stComm_tests`
- Example binary: `build/examples/simple_usage`

## 3. Try Examples (Recommended!)

Run the comprehensive example showing all high-level APIs:

```bash
# Load environment
source env.sh

# Run with 4 MPI processes
mpirun -np 4 ./build/examples/simple_usage
```

This example demonstrates:
- `allgather_vec()` - Simple gathering with equal counts
- `allgatherv_auto()` - Variable-length gather with auto displacement
- `allgatherv_vec()` - Vector-based variable-length gather
- `alltoallv_auto()` - Personalized exchange with auto displacement
- `alltoallv_vec()` - Vector-based personalized exchange

## 4. Run Tests

```bash
# Run with 2 MPI processes
./run_tests.sh

# Run with more processes
./run_tests.sh -np 4

# Run specific tests
./run_tests.sh -f "MPICommTest.SendRecv*"
```

## 5. Use in Your Code

### Simple MPI Example

Create `example.cpp`:

```cpp
#include "stComm/stComm.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // Initialize
    stComm::MPIComm::initialize(&argc, &argv);

    stComm::MPIComm comm;
    int rank = comm.getRank();
    int size = comm.getSize();

    std::cout << "Hello from rank " << rank << " of " << size << std::endl;

    if (size >= 2) {
        std::vector<float> data(100);

        if (rank == 0) {
            // Fill data
            for (int i = 0; i < 100; ++i) {
                data[i] = i * 1.5f;
            }

            // Send asynchronously
            auto req = comm.send(data.data(), data.size(), 1, 0);
            std::cout << "Rank 0: Sent data, doing other work..." << std::endl;

            // Do other work here...

            // Wait for send to complete
            req->wait();
            std::cout << "Rank 0: Send completed" << std::endl;

        } else if (rank == 1) {
            // Receive asynchronously
            auto req = comm.recv(data.data(), data.size(), 0, 0);
            std::cout << "Rank 1: Receiving data..." << std::endl;

            // Wait for receive
            req->wait();
            std::cout << "Rank 1: Received " << data.size()
                      << " elements, first=" << data[0] << std::endl;
        }
    }

    // Finalize
    stComm::MPIComm::finalize();
    return 0;
}
```

### Compile and Run

```bash
# Load environment
source env.sh

# Compile
mpicxx -std=c++17 example.cpp \
    -I./include \
    -L./build/lib \
    -lstComm \
    -o example

# Run with 2 processes
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
mpirun -np 2 ./example
```

### NCCL Example (GPU)

```cpp
#include "stComm/stComm.h"
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);
    stComm::MPIComm mpi_comm;

    int rank = mpi_comm.getRank();
    int size = mpi_comm.getSize();
    int device_id = rank % 4;

    // Setup NCCL
    ncclUniqueId nccl_id;
    if (rank == 0) {
        nccl_id = stComm::NCCLComm::getUniqueId();
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    stComm::NCCLComm nccl_comm;
    nccl_comm.initialize(rank, size, device_id, nccl_id);

    // Allocate GPU memory
    const int N = 1000;
    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    // Initialize with rank value
    std::vector<float> h_data(N, rank * 1.0f);
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float),
               cudaMemcpyHostToDevice);

    // AllReduce (sum across all GPUs)
    auto req = nccl_comm.allreduce(d_data, d_result, N, ncclSum);
    req->wait();

    // Copy result back
    std::vector<float> h_result(N);
    cudaMemcpy(h_result.data(), d_result, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "Rank " << rank << ": AllReduce sum = "
              << h_result[0] << std::endl;

    cudaFree(d_data);
    cudaFree(d_result);

    stComm::MPIComm::finalize();
    return 0;
}
```

Compile with CUDA:

```bash
nvcc -std=c++17 example_nccl.cpp \
    -I./include \
    -L./build/lib \
    -L${MPI_HOME}/lib \
    -lstComm -lmpi -lnccl \
    -o example_nccl

mpirun -np 2 ./example_nccl
```

## 6. API Overview

### Core Classes

- **MPIComm**: MPI-based communication (CPU)
  - **Point-to-point**: `send<T>()`, `recv<T>()`
  - **Collectives (low-level)**: `allgatherv<T>()`, `alltoallv<T>()`
  - **Collectives (auto)**: `allgatherv_auto<T>()`, `alltoallv_auto<T>()` - **No manual displacement!**
  - **Collectives (vector)**: `allgatherv_vec<T>()`, `alltoallv_vec<T>()` - **Returns result directly!**
  - **Simple collectives**: `allgather<T>()`, `allgather_vec<T>()`, `allreduce<T>()`

- **NCCLComm**: NCCL-based communication (GPU)
  - **Point-to-point**: `send<T>()`, `recv<T>()`
  - **Collectives (low-level)**: `allgatherv<T>()`, `alltoallv<T>()`
  - **Collectives (auto)**: `allgatherv_auto<T>()`, `alltoallv_auto<T>()` - **No manual displacement!**
  - **Simple collectives**: `allgather<T>()`, `allreduce<T>()`

- **Request**: Async operation handle
  - `wait()`: Block until complete
  - `test()`: Check completion

- **Utils**: Helper functions
  - `calculate_displacements()`: Auto displacement calculation
  - `total_size()`: Calculate total buffer size from counts
  - `CommBuffer<T>`: RAII buffer management

### Three Levels of API

**1. Vector API (Easiest)** - For maximum convenience
```cpp
// Returns result, handles everything
auto result = comm.allgatherv_vec(sendbuf, recvcounts);
auto result = comm.alltoallv_vec(sendbuf, sendcounts, recvcounts);
```

**2. Auto API (Medium)** - Auto displacement, you manage buffers
```cpp
// No manual displacement calculation needed
comm.allgatherv_auto(sendbuf, count, recvbuf, recvcounts);
comm.alltoallv_auto(sendbuf, sendcounts, recvbuf, recvcounts);
```

**3. Low-level API (Full control)** - When you need maximum performance
```cpp
// Full control, manual displacement
comm.allgatherv(sendbuf, count, recvbuf, recvcounts, displs);
comm.alltoallv(sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls);
```

### Supported Types

All primitive types: `int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `float`, `double`, `char`, `int`, `long`, etc.

## 7. Common Patterns

### Pattern 1: Variable-Length Gather (High-Level API)

```cpp
// Each rank has different amount of data
std::vector<int> my_data(rank + 1);
for (size_t i = 0; i < my_data.size(); ++i) {
    my_data[i] = rank * 100 + i;
}

// Define how much each rank sends
std::vector<int> recvcounts(size);
for (int i = 0; i < size; ++i) {
    recvcounts[i] = i + 1;
}

// Option 1: Vector API (easiest!)
auto result = comm.allgatherv_vec(my_data, recvcounts);
// Done! result contains all gathered data

// Option 2: Auto API (if you need buffer control)
int total = stComm::Utils::total_size(recvcounts);
std::vector<int> recvbuf(total);
auto req = comm.allgatherv_auto(my_data.data(), my_data.size(),
                                recvbuf.data(), recvcounts.data());
req->wait();
```

### Pattern 2: Personalized All-to-All (High-Level API)

```cpp
// Each rank sends different amounts to each other rank
std::vector<int> sendcounts(size);
std::vector<int> recvcounts(size);
for (int i = 0; i < size; ++i) {
    sendcounts[i] = rank + 1;  // Send (rank+1) items to each rank
    recvcounts[i] = i + 1;     // Receive (i+1) items from rank i
}

std::vector<int> sendbuf(stComm::Utils::total_size(sendcounts));
// Fill sendbuf...

// Option 1: Vector API (easiest!)
auto result = comm.alltoallv_vec(sendbuf, sendcounts, recvcounts);

// Option 2: Auto API
std::vector<int> recvbuf(stComm::Utils::total_size(recvcounts));
auto req = comm.alltoallv_auto(sendbuf.data(), sendcounts.data(),
                               recvbuf.data(), recvcounts.data());
req->wait();
```

### Pattern 3: Async Send/Recv with Overlap

```cpp
auto req = comm.send(data, count, dest);
// Do computation here while send is in progress
compute_something_else();
// Wait only when needed
req->wait();
```

### Pattern 4: Multiple Concurrent Operations

```cpp
std::vector<stComm::RequestPtr> requests;

for (int i = 0; i < N; ++i) {
    auto req = comm.send(buffers[i], sizes[i], dests[i], i);
    requests.push_back(req);
}

// Wait for all
for (auto& req : requests) {
    req->wait();
}
```

### Pattern 5: Using Utility Functions

```cpp
// Calculate displacements automatically
std::vector<int> counts = {10, 20, 30, 40};
auto displs = stComm::Utils::calculate_displacements(counts);
// Result: [0, 10, 30, 60]

// Calculate total buffer size
int total = stComm::Utils::total_size(counts);
// Result: 100

// RAII buffer management
stComm::CommBuffer<double> buffer(counts);
// buffer.data() ready to use
// Automatically freed when out of scope
```

## 8. Troubleshooting

### Build fails with "NCCL not found"
```bash
# Set NCCL_HOME in env.sh
export NCCL_HOME="/path/to/nccl"
```

### Tests fail with "symbol not found"
```bash
# Ensure library path is set
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
```

### CUDA out of memory
```bash
# Reduce data size or increase GPU memory
# Check with: nvidia-smi
```

### MPI tests hang
```bash
# Ensure you have enough processes
# Some tests require at least 2 processes
mpirun -np 2 ./build/tests/stComm_tests
```

## Next Steps

- Read full documentation in `README.md`
- Explore test examples in `tests/`
- Check header files in `include/stComm/` for complete API
- Integrate into your CMake project

## Getting Help

- Check `README.md` for detailed documentation
- Review test cases in `tests/` for usage examples
- Open an issue if you encounter problems
