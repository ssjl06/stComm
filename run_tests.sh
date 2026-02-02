#!/bin/bash
# stComm test runner script

set -e

# Source environment if exists
if [ -f "env.sh" ]; then
    source env.sh
fi

# Default number of MPI processes
NP=2
TEST_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -np)
            NP="$2"
            shift 2
            ;;
        -f|--filter)
            TEST_FILTER="--gtest_filter=$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-np NUM_PROCESSES] [-f TEST_FILTER]"
            echo ""
            echo "Options:"
            echo "  -np NUM        Number of MPI processes (default: 2)"
            echo "  -f FILTER      Google Test filter pattern"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# Check if test binary exists
if [ ! -f "build/tests/stComm_tests" ]; then
    echo "Error: Test binary not found. Run ./build.sh first."
    exit 1
fi

# Set library path
export LD_LIBRARY_PATH=build/lib:${LD_LIBRARY_PATH}

# Run tests with MPI
echo "=========================================="
echo "Running stComm tests with ${NP} MPI processes"
if [ -n "$TEST_FILTER" ]; then
    echo "Filter: $TEST_FILTER"
fi
echo "=========================================="
echo ""

mpirun -np ${NP} ./build/tests/stComm_tests ${TEST_FILTER}

echo ""
echo "=========================================="
echo "Tests completed"
echo "=========================================="
