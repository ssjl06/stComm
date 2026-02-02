# Examples for stComm library

# Simple usage example
add_executable(simple_usage simple_usage.cpp)
target_link_libraries(simple_usage stComm)
target_include_directories(simple_usage PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Unified interface example
add_executable(unified_interface unified_interface.cpp)
target_link_libraries(unified_interface stComm)
target_include_directories(unified_interface PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Install examples to build/examples
install(TARGETS simple_usage unified_interface
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/examples)
