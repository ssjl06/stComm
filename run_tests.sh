#!/bin/bash
set -e

# stComm Test Runner Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TEST_BIN="${BUILD_DIR}/tests/stComm_tests"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== stComm Test Runner ===${NC}"

# Load environment
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    source "${SCRIPT_DIR}/env.sh"
fi

# Check if tests are built
if [ ! -f "${TEST_BIN}" ]; then
    echo -e "${RED}Error: Test executable not found${NC}"
    echo -e "Please build the project first: ./build.sh"
    exit 1
fi

# Parse arguments
NUM_PROCS=2
TEST_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -np|--num-procs)
            NUM_PROCS="$2"
            shift 2
            ;;
        -f|--filter)
            TEST_FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -np, --num-procs N  Number of MPI processes (default: 2)"
            echo "  -f, --filter PATTERN  Run only tests matching pattern"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set library path
export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH}"

# Run tests
echo -e "${GREEN}Running tests with ${NUM_PROCS} MPI processes...${NC}"
echo -e "Test binary: ${TEST_BIN}"
echo -e ""

GTEST_ARGS=""
if [ -n "${TEST_FILTER}" ]; then
    GTEST_ARGS="--gtest_filter=${TEST_FILTER}"
    echo -e "${YELLOW}Filter: ${TEST_FILTER}${NC}"
fi

if command -v mpirun &> /dev/null; then
    mpirun -np ${NUM_PROCS} "${TEST_BIN}" ${GTEST_ARGS}
else
    echo -e "${YELLOW}Warning: mpirun not found, running without MPI${NC}"
    "${TEST_BIN}" ${GTEST_ARGS}
fi

echo -e ""
echo -e "${GREEN}=== Tests Complete ===${NC}"
