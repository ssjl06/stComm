#!/bin/bash
set -e

# stComm Build Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== stComm Build System ===${NC}"

# Load environment configuration
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    echo -e "${GREEN}Loading environment from env.sh...${NC}"
    source "${SCRIPT_DIR}/env.sh"
else
    echo -e "${YELLOW}Warning: env.sh not found. Using system defaults.${NC}"
    echo -e "${YELLOW}Copy env.sh.example to env.sh and configure paths if needed.${NC}"
fi

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN_BUILD=false
NUM_JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --debug        Build in Debug mode (default: Release)"
            echo "  -c, --clean        Clean build directory before building"
            echo "  -j, --jobs N       Number of parallel jobs (default: nproc)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

# Configure
echo -e "${GREEN}Configuring with CMake (${BUILD_TYPE})...${NC}"
cd "${BUILD_DIR}"

${CMAKE:-cmake} \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${CC:-gcc}" \
    -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
    -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}/install" \
    "${SCRIPT_DIR}"

# Build
echo -e "${GREEN}Building with ${NUM_JOBS} parallel jobs...${NC}"
${CMAKE:-cmake} --build . -j ${NUM_JOBS}

# Check build results
if [ -f "${BUILD_DIR}/lib/libstComm.so" ]; then
    echo -e "${GREEN}=== Build Successful ===${NC}"
    echo -e "Library: ${BUILD_DIR}/lib/libstComm.so"
    ls -lh "${BUILD_DIR}/lib/libstComm.so"
else
    echo -e "${RED}=== Build Failed ===${NC}"
    echo -e "${RED}Library not found at ${BUILD_DIR}/lib/libstComm.so${NC}"
    exit 1
fi

if [ -f "${BUILD_DIR}/tests/stComm_tests" ]; then
    echo -e "Tests:   ${BUILD_DIR}/tests/stComm_tests"
    ls -lh "${BUILD_DIR}/tests/stComm_tests"
else
    echo -e "${YELLOW}Warning: Test executable not found${NC}"
fi

echo -e ""
echo -e "${GREEN}Build artifacts:${NC}"
echo -e "  Libraries: ${BUILD_DIR}/lib/"
echo -e "  Tests:     ${BUILD_DIR}/tests/"
echo -e ""
echo -e "${GREEN}To run tests:${NC}"
echo -e "  cd ${BUILD_DIR}/tests"
echo -e "  mpirun -np 2 ./stComm_tests"
echo -e ""
echo -e "${GREEN}To install:${NC}"
echo -e "  cd ${BUILD_DIR}"
echo -e "  make install"
