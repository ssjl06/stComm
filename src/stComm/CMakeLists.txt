# stComm library build configuration

# Source files
set(STCOMM_SOURCES
    request.cpp
    mpi_comm.cpp
    nccl_comm.cpp
    communicator.cpp
    stComm.cpp
)

# Create shared library
add_library(stComm SHARED ${STCOMM_SOURCES})

# Set library properties
set_target_properties(stComm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(stComm
    PUBLIC
        MPI::MPI_CXX
        CUDA::cudart
        ${NCCL_LIBRARY}
)

# Include directories
target_include_directories(stComm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Installation rules
install(TARGETS stComm
    EXPORT stCommTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/stComm
    DESTINATION include
)

# Export targets
install(EXPORT stCommTargets
    FILE stCommTargets.cmake
    NAMESPACE stComm::
    DESTINATION lib/cmake/stComm
)

# Generate config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/stCommConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/stCommConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/stCommConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/stCommConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/stCommConfigVersion.cmake"
    DESTINATION lib/cmake/stComm
)
