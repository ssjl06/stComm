# stComm library build configuration

# Source files
set(STCOMM_SOURCES
    mpi_comm.cpp
    nccl_comm.cpp
)

# Create shared library
add_library(stComm SHARED ${STCOMM_SOURCES})

# Set library properties
set_target_properties(stComm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
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
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
