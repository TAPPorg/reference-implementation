cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

# see https://semver.org/
set (TAPP_CUTENSOR_MAJOR_VERSION 0)
set (TAPP_CUTENSOR_MINOR_VERSION 5)
set (TAPP_CUTENSOR_PATCH_VERSION 0)
set (TAPP_CUTENSOR_PRERELEASE_ID )
set (TAPP_CUTENSOR_BUILD_ID )

set(TAPP_CUTENSOR_VERSION "${TAPP_CUTENSOR_MAJOR_VERSION}.${TAPP_CUTENSOR_MINOR_VERSION}.${TAPP_CUTENSOR_PATCH_VERSION}")
if (TAPP_CUTENSOR_PRERELEASE_ID)
  set(TAPP_CUTENSOR_EXT_VERSION "${TAPP_CUTENSOR_VERSION}-${TAPP_CUTENSOR_PRERELEASE_ID}")
else(TAPP_CUTENSOR_PRERELEASE_ID)
  set(TAPP_CUTENSOR_EXT_VERSION "${TAPP_CUTENSOR_VERSION}")
endif(TAPP_CUTENSOR_PRERELEASE_ID)
if (TAPP_CUTENSOR_BUILD_ID)
  set(TAPP_CUTENSOR_EXT_VERSION "${TAPP_CUTENSOR_EXT_VERSION}+${TAPP_CUTENSOR_BUILD_ID}")
endif(TAPP_CUTENSOR_BUILD_ID)

# Extract the git revision tag information
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
          COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_VARIABLE TAPP_CUTENSOR_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
          TAPP_CUTENSOR_REVISION "${TAPP_CUTENSOR_REVISION}")
else()
  set(TAPP_CUTENSOR_REVISION "unknown")
endif()

project(tapp-cutensor
        VERSION ${TAPP_CUTENSOR_VERSION}
        DESCRIPTION "TAPP implementation backed by cuTENSOR"
        LANGUAGES CXX
        HOMEPAGE_URL "https://github.com/TAPPOrg/")

include(GNUInstallDirs)

set(TAPP_CUTENSOR_INSTALL_BINDIR "bin"
        CACHE PATH "TAPP CUTENSOR binary install directory")
set(TAPP_CUTENSOR_INSTALL_INCLUDEDIR "include"
        CACHE PATH "TAPP CUTENSOR INCLUDE install directory")
set(TAPP_CUTENSOR_INSTALL_LIBDIR "lib"
        CACHE PATH "TAPP CUTENSOR LIB install directory")
set(TAPP_CUTENSOR_INSTALL_DATADIR "share/mpqc/${TAPP_CUTENSOR_EXT_VERSION}/data"
        CACHE PATH "TAPP CUTENSOR DATA install directory")
set(TAPP_CUTENSOR_INSTALL_DOCDIR "share/tapp/${TAPP_CUTENSOR_EXT_VERSION}/doc"
        CACHE PATH "TAPP CUTENSOR DOC install directory")
set(TAPP_CUTENSOR_INSTALL_CMAKEDIR "lib/cmake/mpqc"
        CACHE PATH "TAPP CUTENSOR CMAKE install directory")

# Expects the tapp-api and cutensor::cutensor targets (the latter found in the
# top-level CMakeLists.txt) to be available.

add_library(tapp-cutensor SHARED)
set_property(TARGET tapp-cutensor PROPERTY EXPORT_NAME cutensor)
add_library(tapp::cutensor ALIAS tapp-cutensor)

target_sources(
  tapp-cutensor
  PRIVATE
    src/attributes.cpp
    src/datatype.cpp
    src/error.cpp
    src/executor.cpp
    src/product.cpp
    src/permute.cpp
    src/runtime.cpp
    src/status.cpp
    src/tensor.cpp
    src/util.cpp
    src/internal.h
    include/tapp_cutensor.h
  )

set_property(
   TARGET tapp-cutensor
   PROPERTY
     CXX_STANDARD 20
     CXX_STANDARD_REQUIRED YES
     CXX_EXTENSIONS NO
     POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
  tapp-cutensor
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  tapp-cutensor
  PUBLIC
    tapp-api
  PRIVATE
    cutensor::cutensor
    CUDA::cudart
)

if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  target_link_options(tapp-cutensor PRIVATE "-undefined;dynamic_lookup")
endif()

install(TARGETS tapp-cutensor EXPORT tapp
  COMPONENT cutensor)
