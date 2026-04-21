cmake_minimum_required(VERSION 3.25)

# ── CUDA architectures (must precede project() for CMP0104) ──────────────────
# project(LANGUAGES CUDA) writes CMAKE_CUDA_ARCHITECTURES into the cache during
# compiler detection; a subsequent set(...CACHE STRING) without FORCE is silently
# ignored, leaving "native"/empty on headless CI runners (→ nvcc defaults to
# sm_52 which is < 6.0 and fails MatX's compile-time check).
# Override via:  cmake -DMATX_CUDA_ARCH="80;86;89;90"
if(DEFINED MATX_CUDA_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${MATX_CUDA_ARCH})
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

project(MatxNative
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

# ── C++ / CUDA standards ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

## ── CUDA architectures ────────────────────────────────────────────────────────
## Set MATX_CUDA_ARCH on the command line to override, e.g.:
##   cmake -DMATX_CUDA_ARCH="80;86;89;90" ..
#if(NOT DEFINED MATX_CUDA_ARCH)
#    # Auto-detect from the host GPU; fall back to a broad set.
#    include(FindCUDA OPTIONAL)
#    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING
#        "CUDA GPU architectures to compile for")
#else()
#    set(CMAKE_CUDA_ARCHITECTURES ${MATX_CUDA_ARCH})
#endif()

# ── Platform detection ────────────────────────────────────────────────────────
if(WIN32)
    set(MATX_PLATFORM_WINDOWS TRUE)
elseif(APPLE)
    set(MATX_PLATFORM_MACOS TRUE)
    message(WARNING "macOS: CUDA GPU kernels unavailable. Only host executor path is compiled.")
else()
    set(MATX_PLATFORM_LINUX TRUE)
endif()

# ── Find CUDA Toolkit ─────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── MatX (header-only) ───────────────────────────────────────────────────────
# Assume the CMakeLists.txt of the parent MatX repo is one directory up.
# Adjust MATX_ROOT if building standalone.
if(NOT DEFINED MATX_ROOT)
    get_filename_component(MATX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/MatX" ABSOLUTE)
endif()

# If matx was installed via find_package:
find_package(matx CONFIG QUIET HINTS "${MATX_ROOT}/build" "${MATX_ROOT}")

if(NOT matx_FOUND)
    # Header-only fallback – just add the include directory.
    if(NOT EXISTS "${MATX_ROOT}/include/matx.h")
        message(FATAL_ERROR "Cannot find MatX headers. Set MATX_ROOT to the MatX repo root.")
    endif()
    add_library(matx INTERFACE)
    target_include_directories(matx INTERFACE "${MATX_ROOT}/include")
    message(STATUS "Using MatX from ${MATX_ROOT}/include (header-only)")
endif()

# ── Source files ──────────────────────────────────────────────────────────────
set(NATIVE_SOURCES
    src/matx_core.cu
    src/matx_elementwise.cu
    src/matx_generators.cu
    src/matx_fft.cu
    src/matx_linalg.cu
    src/matx_signal.cu
)

# ── Shared library ────────────────────────────────────────────────────────────
add_library(matxnative SHARED ${NATIVE_SOURCES})

target_compile_definitions(matxnative PRIVATE MATX_NATIVE_EXPORTS)

target_include_directories(matxnative PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(matxnative PRIVATE
    matx
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
    CUDA::cusolver
    CUDA::cusparse
)

# cuBLAS / cuFFT / cuSolver are optional but almost always present.
foreach(_lib cublas cufft cusolver cusparse)
    find_library(_lib_path ${_lib} HINTS ${CUDAToolkit_LIBRARY_DIR})
    if(_lib_path)
        target_link_libraries(matxnative PRIVATE CUDA::${_lib})
    endif()
    unset(_lib_path CACHE)
endforeach()

# ── CUDA compile flags ────────────────────────────────────────────────────────
target_compile_options(matxnative PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -lineinfo
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-G -g>
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wno-unused-parameter>
        $<$<CXX_COMPILER_ID:MSVC>:/W3>
    >
)

# ── Windows: export all symbols automatically when building shared lib ────────
if(MATX_PLATFORM_WINDOWS)
    set_target_properties(matxnative PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
endif()

# ── Output naming ─────────────────────────────────────────────────────────────
set_target_properties(matxnative PROPERTIES
    OUTPUT_NAME   "matxnative"
    VERSION       "${PROJECT_VERSION}"
    SOVERSION     "1"
    POSITION_INDEPENDENT_CODE ON
)

# ── Install rules ─────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS matxnative
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES include/matx_api.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
