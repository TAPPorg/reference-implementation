cmake_minimum_required(VERSION 3.18)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

# see https://semver.org/
set (TAPP_CUBLAS_MAJOR_VERSION 0)
set (TAPP_CUBLAS_MINOR_VERSION 5)
set (TAPP_CUBLAS_PATCH_VERSION 0)
set (TAPP_CUBLAS_PRERELEASE_ID )
set (TAPP_CUBLAS_BUILD_ID )

set(TAPP_CUBLAS_VERSION "${TAPP_CUBLAS_MAJOR_VERSION}.${TAPP_CUBLAS_MINOR_VERSION}.${TAPP_CUBLAS_PATCH_VERSION}")
if (TAPP_CUBLAS_PRERELEASE_ID)
  set(TAPP_CUBLAS_EXT_VERSION "${TAPP_CUBLAS_VERSION}-${TAPP_CUBLAS_PRERELEASE_ID}")
else(TAPP_CUBLAS_PRERELEASE_ID)
  set(TAPP_CUBLAS_EXT_VERSION "${TAPP_CUBLAS_VERSION}")
endif(TAPP_CUBLAS_PRERELEASE_ID)
if (TAPP_CUBLAS_BUILD_ID)
  set(TAPP_CUBLAS_EXT_VERSION "${TAPP_CUBLAS_EXT_VERSION}+${TAPP_CUBLAS_BUILD_ID}")
endif(TAPP_CUBLAS_BUILD_ID)

# Extract the git revision tag information
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
          COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_VARIABLE TAPP_CUBLAS_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
          TAPP_CUBLAS_REVISION "${TAPP_CUBLAS_REVISION}")
else()
  set(TAPP_CUBLAS_REVISION "unknown")
endif()

project(tapp-cublas
        VERSION ${TAPP_CUBLAS_VERSION}
        DESCRIPTION "TAPP implementation backed by cuBLAS (TTGT) with optional Ozaki emulation"
        LANGUAGES CXX CUDA
        HOMEPAGE_URL "https://github.com/TAPPOrg/")

# Pass include directories to nvcc on the command line instead of in a response file, so they show up in
# compile_commands.json and editors can resolve the includes of the .cu files
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

include(GNUInstallDirs)

set(TAPP_CUBLAS_INSTALL_BINDIR "bin"
        CACHE PATH "TAPP CUBLAS binary install directory")
set(TAPP_CUBLAS_INSTALL_INCLUDEDIR "include"
        CACHE PATH "TAPP CUBLAS INCLUDE install directory")
set(TAPP_CUBLAS_INSTALL_LIBDIR "lib"
        CACHE PATH "TAPP CUBLAS LIB install directory")
set(TAPP_CUBLAS_INSTALL_DATADIR "share/mpqc/${TAPP_CUBLAS_EXT_VERSION}/data"
        CACHE PATH "TAPP CUBLAS DATA install directory")
set(TAPP_CUBLAS_INSTALL_DOCDIR "share/tapp/${TAPP_CUBLAS_EXT_VERSION}/doc"
        CACHE PATH "TAPP CUBLAS DOC install directory")
set(TAPP_CUBLAS_INSTALL_CMAKEDIR "lib/cmake/mpqc"
        CACHE PATH "TAPP CUBLAS CMAKE install directory")

# Expects the tapp-api and cutt targets, and with TAPP_OZAKI_EMULATION_VERSION 2 the
# gemmul8 target, to be available (set up in the top-level CMakeLists.txt).

add_library(tapp-cublas SHARED)
set_property(TARGET tapp-cublas PROPERTY EXPORT_NAME cublas)
add_library(tapp::cublas ALIAS tapp-cublas)

target_sources(
  tapp-cublas
  PRIVATE
    src/attributes.cpp
    src/datatype.cpp
    src/error.cpp
    src/executor.cpp
    src/product.cu
    src/runtime.cpp
    src/status.cpp
    src/tensor.cpp
    src/internal.h
    include/tapp_cublas.h
  )

set_property(
   TARGET tapp-cublas
   PROPERTY
     CXX_STANDARD 20
     CXX_STANDARD_REQUIRED YES
     CXX_EXTENSIONS NO
     CUDA_STANDARD 20
     CUDA_STANDARD_REQUIRED YES
     CUDA_EXTENSIONS NO
     POSITION_INDEPENDENT_CODE ON
)

# Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> to target a specific GPU (e.g. 80 for
# A100, 90 for H100/Grace-Hopper). When unset, fall back to a portable list
# rather than "native" so the library can be built on a GPU-less login node.
# Volta (70) is left out because CUDA 13 dropped it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET tapp-cublas PROPERTY CUDA_ARCHITECTURES "80;90")
endif()

target_compile_definitions(
  tapp-cublas
  PRIVATE
    OZAKI_EMULATION_VERSION=${TAPP_OZAKI_EMULATION_VERSION}
)

target_include_directories(
  tapp-cublas
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  tapp-cublas
  PUBLIC
    tapp-api
  PRIVATE
    CUDA::cudart
    CUDA::cublas
    cutt
)

if(TAPP_OZAKI_EMULATION_VERSION STREQUAL "2")
  target_compile_definitions(
    tapp-cublas
    PRIVATE
      OZAKI_II_COMPUTE_CAPABILITY=${OZAKI_II_COMPUTE_CAPABILITY}
  )
  target_link_libraries(tapp-cublas PRIVATE gemmul8)
endif()

install(TARGETS tapp-cublas EXPORT tapp
  COMPONENT cublas)
