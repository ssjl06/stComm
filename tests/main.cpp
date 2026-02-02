/**
 * @file main.cpp
 * @brief Main entry point for stComm tests
 */

#include <gtest/gtest.h>
#include "stComm/stComm.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    stComm::MPIComm::initialize(&argc, &argv);

    int result = RUN_ALL_TESTS();

    stComm::MPIComm::finalize();
    return result;
}
