# stComm Installation Guide

## Prerequisites

Before installing stComm, ensure you have:
- GCC/G++ with C++17 support
- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher
- OpenMPI or MPICH
- NCCL 2.0 or higher

## Installation Steps

### 1. Configure Environment

```bash
cp env.sh.example env.sh
vim env.sh  # Edit paths for your system
source env.sh
```

### 2. Build

```bash
./build.sh
```

This will create:
- `build/lib/libstComm.so` - Shared library
- `build/tests/stComm_tests` - Test binary
- `build/examples/` - Example binaries

### 3. Test (Optional)

```bash
./run_tests.sh
```

### 4. Install

Choose one of the following installation methods:

#### Option A: System-wide Installation (requires sudo)

```bash
sudo ./install.sh
```

This installs to `/usr/local/`:
- Headers: `/usr/local/include/stComm/`
- Library: `/usr/local/lib/libstComm.so`
- CMake config: `/usr/local/lib/cmake/stComm/`

#### Option B: User-local Installation (recommended)

```bash
./install.sh $HOME/local/stComm
```

This installs to `$HOME/local/stComm/`:
- Headers: `$HOME/local/stComm/include/stComm/`
- Library: `$HOME/local/stComm/lib/libstComm.so`
- CMake config: `$HOME/local/stComm/lib/cmake/stComm/`

**Important:** Add to your `~/.bashrc` or `~/.bash_profile`:

```bash
export CMAKE_PREFIX_PATH=$HOME/local/stComm:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/stComm/lib:$LD_LIBRARY_PATH
```

#### Option C: Custom Location

```bash
./install.sh /path/to/custom/location
```

Remember to set `CMAKE_PREFIX_PATH` and `LD_LIBRARY_PATH` accordingly.

## Verification

After installation, verify by compiling a simple test program:

```bash
# Create test directory
mkdir -p ~/test_stcomm && cd ~/test_stcomm

# Create test program
cat > test.cpp << 'EOF'
#include "stComm/stComm.h"
#include <iostream>

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);
    stComm::MPIComm comm;

    std::cout << "Hello from rank " << comm.getRank()
              << " of " << comm.getSize() << std::endl;

    stComm::MPIComm::finalize();
    return 0;
}
EOF

# Create CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(test_stcomm LANGUAGES CXX)

find_package(stComm REQUIRED)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE stComm::stComm)
EOF

# Build
mkdir build && cd build
cmake ..
make

# Run
mpirun -np 2 ./test
```

Expected output:
```
Hello from rank 0 of 2
Hello from rank 1 of 2
```

## Uninstallation

To uninstall stComm:

```bash
# From build directory
cd build
cat install_manifest.txt | sudo xargs rm -f

# Or manually remove:
sudo rm -rf /usr/local/include/stComm
sudo rm -f /usr/local/lib/libstComm.so*
sudo rm -rf /usr/local/lib/cmake/stComm
```

## Troubleshooting

### CMake can't find stComm

Ensure `CMAKE_PREFIX_PATH` is set:
```bash
export CMAKE_PREFIX_PATH=/path/to/stComm/install:$CMAKE_PREFIX_PATH
```

### Runtime library not found

Ensure `LD_LIBRARY_PATH` is set:
```bash
export LD_LIBRARY_PATH=/path/to/stComm/install/lib:$LD_LIBRARY_PATH
```

Or add to `/etc/ld.so.conf.d/stcomm.conf`:
```
/path/to/stComm/install/lib
```
Then run: `sudo ldconfig`

### NCCL not found during build

Set `NCCL_HOME` before building:
```bash
export NCCL_HOME=/path/to/nccl
./build.sh
./install.sh
```

## Advanced: Building from Installed Library

See `examples/` directory for complete examples. A minimal example:

**main.cpp:**
```cpp
#include "stComm/stComm.h"
#include <vector>

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);
    stComm::MPIComm comm;

    int rank = comm.getRank();
    int size = comm.getSize();

    // Ring communication
    std::vector<int> data(1000, rank);
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    auto send_req = comm.send(data.data(), 1000, next, 0);
    auto recv_req = comm.recv(data.data(), 1000, prev, 0);

    send_req->wait();
    recv_req->wait();

    stComm::MPIComm::finalize();
    return 0;
}
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(MyApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(stComm REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE stComm::stComm)
```

**Build and run:**
```bash
mkdir build && cd build
cmake ..
make
mpirun -np 4 ./my_app
```
