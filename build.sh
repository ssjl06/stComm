#!/bin/bash
# stComm build script

set -e

# Source environment if exists
if [ -f "env.sh" ]; then
    source env.sh
    echo "Loaded environment from env.sh"
else
    echo "Warning: env.sh not found. Using system defaults."
    echo "Copy env.sh.example to env.sh and configure paths."
fi

# Parse arguments
BUILD_TYPE="Release"
CLEAN_BUILD=false
NUM_JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -j)
            NUM_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug] [--clean] [-j NUM_JOBS]"
            exit 1
            ;;
    esac
done

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring with CMake (${BUILD_TYPE})..."
${CMAKE:-cmake} .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

# Build
echo "Building with ${NUM_JOBS} jobs..."
${CMAKE:-cmake} --build . -j ${NUM_JOBS}

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Library: build/lib/libstComm.so"
echo "Tests: build/tests/stComm_tests"
echo ""
echo "To run tests:"
echo "  ./run_tests.sh"
echo "=========================================="
