# Examples for stComm library

# Simple usage example
add_executable(simple_usage simple_usage.cpp)
target_link_libraries(simple_usage stComm)
target_include_directories(simple_usage PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Install examples to build/examples
install(TARGETS simple_usage
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/examples)
