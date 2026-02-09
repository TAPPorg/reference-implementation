cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

# see https://semver.org/
set (CUTENSOR_BINDINGS_MAJOR_VERSION 0)
set (CUTENSOR_BINDINGS_MINOR_VERSION 5)
set (CUTENSOR_BINDINGS_PATCH_VERSION 0)
set (CUTENSOR_BINDINGS_PRERELEASE_ID )
set (CUTENSOR_BINDINGS_BUILD_ID )

set(CUTENSOR_BINDINGS_VERSION "${CUTENSOR_BINDINGS_MAJOR_VERSION}.${CUTENSOR_BINDINGS_MINOR_VERSION}.${CUTENSOR_BINDINGS_PATCH_VERSION}")
if (CUTENSOR_BINDINGS_PRERELEASE_ID)
  set(CUTENSOR_BINDINGS_EXT_VERSION "${CUTENSOR_BINDINGS_VERSION}-${CUTENSOR_BINDINGS_PRERELEASE_ID}")
else(CUTENSOR_BINDINGS_PRERELEASE_ID)
  set(CUTENSOR_BINDINGS_EXT_VERSION "${CUTENSOR_BINDINGS_VERSION}")
endif(CUTENSOR_BINDINGS_PRERELEASE_ID)
if (CUTENSOR_BINDINGS_BUILD_ID)
  set(CUTENSOR_BINDINGS_EXT_VERSION "${CUTENSOR_BINDINGS_EXT_VERSION}+${CUTENSOR_BINDINGS_BUILD_ID}")
endif(CUTENSOR_BINDINGS_BUILD_ID)

# Extract the git revision tag information
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
          COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_VARIABLE CUTENSOR_BINDINGS_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
          CUTENSOR_BINDINGS_REVISION "${CUTENSOR_BINDINGS_REVISION}")
else()
  set(CUTENSOR_BINDINGS_REVISION "unknown")
endif()

project(cutensor_bindings
        VERSION ${CUTENSOR_BINDINGS_VERSION}
        DESCRIPTION "TAPP: Tensor Algebra Processing Primitives - cuTensor Bindings"
        LANGUAGES CXX CUDA
        HOMEPAGE_URL "https://github.com/TAPPOrg/")

include(GNUInstallDirs)

set(CUTENSOR_BINDINGS_INSTALL_BINDIR "bin"
        CACHE PATH "CUTENSOR BINDINGS binary install directory")
set(CUTENSOR_BINDINGS_INSTALL_INCLUDEDIR "include"
        CACHE PATH "CUTENSOR BINDINGS INCLUDE install directory")
set(CUTENSOR_BINDINGS_INSTALL_LIBDIR "lib"
        CACHE PATH "CUTENSOR BINDINGS LIB install directory")
set(CUTENSOR_BINDINGS_INSTALL_DATADIR "share/mpqc/${CUTENSOR_BINDINGS_EXT_VERSION}/data"
        CACHE PATH "CUTENSOR BINDINGS DATA install directory")
set(CUTENSOR_BINDINGS_INSTALL_DOCDIR "share/tapp/${CUTENSOR_BINDINGS_EXT_VERSION}/doc"
        CACHE PATH "CUTENSOR BINDINGS DOC install directory")
set(CUTENSOR_BINDINGS_INSTALL_CMAKEDIR "lib/cmake/mpqc"
        CACHE PATH "CUTENSOR BINDINGS CMAKE install directory")

set(CUTENSOR_ROOT "/usr/local/cutensor")
set(CUTENSOR_INCLUDE_DIR "${CUTENSOR_ROOT}/include")
file(GLOB CUTENSOR_VERSIONED_DIRS "${CUTENSOR_ROOT}/lib/[0-9]*")
set(CUTENSOR_LIBRARY_DIR "${CUTENSOR_ROOT}/lib" ${CUTENSOR_VERSIONED_DIRS})

find_library(
    CUTENSOR_LIB
    NAMES cutensor
    PATHS ${CUTENSOR_LIBRARY_DIR}
)

if (NOT CUTENSOR_LIB)
  message(FATAL_ERROR "cuTENSOR library not found. Set CUTENSOR_ROOT correctly.")
else()
  get_filename_component(CUTENSOR_LIBRARY_DIR ${CUTENSOR_LIB} DIRECTORY)
  if(CUTENSOR_LIBRARY_DIR MATCHES "/[0-9]+$")
    get_filename_component(CUTENSOR_LIBRARY_DIR ${CUTENSOR_LIBRARY_DIR} DIRECTORY)
  endif()
  get_filename_component(CUTENSOR_ROOT ${CUTENSOR_LIBRARY_DIR} DIRECTORY)

  set(CUTENSOR_INCLUDE_DIR "${CUTENSOR_ROOT}/include")
endif()

message(STATUS "Found cuTENSOR: ${CUTENSOR_LIB}")
message(STATUS "cuTENSOR include dir: ${CUTENSOR_INCLUDE_DIR}")

add_library(cutensor_bindings SHARED)

target_sources(
  cutensor_bindings
  PRIVATE
    src/attributes.cu
    src/datatype.cu
    src/error.cu
    src/executor.cu
    src/handle.cu
    src/product.cu
    src/tensor.cu
    include/attributes.h
    include/datatype.h
    include/error.h
    include/executor.h
    include/handle.h
    include/product.h
    include/tensor.h

)

set_property(
   TARGET cutensor_bindings
   PROPERTY
     CUDA_SEPARABLE_COMPILATION ON
     POSITION_INDEPENDENT_CODE ON
     CXX_STANDARD 20
     CXX_STANDARD_REQUIRED YES
     CUDA_STANDARD 20
)

set_property(TARGET cutensor_bindings PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(
  cutensor_bindings
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTENSOR_INCLUDE_DIR}
)

target_link_libraries(cutensor_bindings
  PUBLIC
    tapp-api
  PRIVATE
    ${CUTENSOR_LIB}
)

if (APPLE AND CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  target_link_options(cutensor_bindings PRIVATE "-undefined;dynamic_lookup")
endif()