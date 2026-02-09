#!/bin/bash
# stComm installation script

set -e

# Default install prefix
DEFAULT_PREFIX="/usr/local"
INSTALL_PREFIX="${1:-$DEFAULT_PREFIX}"

# Source environment if exists
if [ -f "env.sh" ]; then
    source env.sh
    echo "Loaded environment from env.sh"
fi

echo "=========================================="
echo "Installing stComm to: ${INSTALL_PREFIX}"
echo "=========================================="

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Build directory not found. Running build first..."
    ./build.sh
fi

# Install
cd build
echo "Installing..."
${CMAKE:-cmake} --install . --prefix "${INSTALL_PREFIX}"

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo "Install location: ${INSTALL_PREFIX}"
echo ""
echo "Headers: ${INSTALL_PREFIX}/include/stComm/"
echo "Library: ${INSTALL_PREFIX}/lib/libstComm.so"
echo "CMake:   ${INSTALL_PREFIX}/lib/cmake/stComm/"
echo ""
echo "To use stComm in your CMake project, add:"
echo "  find_package(stComm REQUIRED)"
echo "  target_link_libraries(your_target PRIVATE stComm::stComm)"
echo ""
echo "You may need to add to your environment:"
echo "  export CMAKE_PREFIX_PATH=${INSTALL_PREFIX}:\$CMAKE_PREFIX_PATH"
echo "  export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:\$LD_LIBRARY_PATH"
echo "=========================================="
