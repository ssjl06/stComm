# enable_compiler_warnings(<target>)
#
# Attaches the project's strict warning set to <target> as PRIVATE options, so
# it never leaks to find_package consumers. C++ TUs only: CUDA is an enabled
# language and nvcc does not accept these flags. GCC-only diagnostics are
# guarded so a Clang build still configures. -Werror follows
# STCOMM_WARNINGS_AS_ERRORS (defined in the root CMakeLists.txt).
function(enable_compiler_warnings tgt)
    set(_cxx "$<COMPILE_LANGUAGE:CXX>")
    set(_gnu "$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>")

    target_compile_options(${tgt} PRIVATE
        "$<${_cxx}:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion>"
        "$<${_cxx}:-Wformat=2;-Wformat-security;-Wvla;-Wshadow;-Wold-style-cast>"
        "$<${_cxx}:-Woverloaded-virtual;-Wnon-virtual-dtor;-Wzero-as-null-pointer-constant>"
        "$<${_cxx}:-Wnull-dereference;-Wcast-qual;-Wmissing-declarations>"
        "$<${_cxx}:-Wmisleading-indentation;-Winit-self;-Wfloat-equal;-Wdouble-promotion>"
        "$<${_cxx}:-Wswitch-enum;-Wundef;-Wpacked;-Wstrict-aliasing=3>"
        "$<${_gnu}:-Wduplicated-cond;-Wduplicated-branches;-Wlogical-op>"
        "$<${_gnu}:-Wuseless-cast;-Wmaybe-uninitialized>"
    )
    if(STCOMM_WARNINGS_AS_ERRORS)
        target_compile_options(${tgt} PRIVATE "$<${_cxx}:-Werror;-Wfatal-errors>")
    endif()
endfunction()
