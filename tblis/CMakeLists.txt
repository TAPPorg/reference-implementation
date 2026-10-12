cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

# see https://semver.org/
set (TAPP_TBLIS_MAJOR_VERSION 0)
set (TAPP_TBLIS_MINOR_VERSION 5)
set (TAPP_TBLIS_PATCH_VERSION 0)
set (TAPP_TBLIS_PRERELEASE_ID )
set (TAPP_TBLIS_BUILD_ID )

set(TAPP_TBLIS_VERSION "${TAPP_TBLIS_MAJOR_VERSION}.${TAPP_TBLIS_MINOR_VERSION}.${TAPP_TBLIS_PATCH_VERSION}")
if (TAPP_TBLIS_PRERELEASE_ID)
  set(TAPP_TBLIS_EXT_VERSION "${TAPP_TBLIS_VERSION}-${TAPP_TBLIS_PRERELEASE_ID}")
else(TAPP_TBLIS_PRERELEASE_ID)
  set(TAPP_TBLIS_EXT_VERSION "${TAPP_TBLIS_VERSION}")
endif(TAPP_TBLIS_PRERELEASE_ID)
if (TAPP_TBLIS_BUILD_ID)
  set(TAPP_TBLIS_EXT_VERSION "${TAPP_TBLIS_EXT_VERSION}+${TAPP_TBLIS_BUILD_ID}")
endif(TAPP_TBLIS_BUILD_ID)

# Extract the git revision tag information
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
          COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_VARIABLE TAPP_TBLIS_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
          TAPP_TBLIS_REVISION "${TAPP_TBLIS_REVISION}")
else()
  set(TAPP_TBLIS_REVISION "unknown")
endif()

project(tapp-tblis
        VERSION ${TAPP_TBLIS_VERSION}
        DESCRIPTION "TAPP implementation backed by TBLIS"
        LANGUAGES C CXX
        HOMEPAGE_URL "https://github.com/TAPPOrg/")

include(GNUInstallDirs)

set(TAPP_TBLIS_INSTALL_BINDIR "bin"
        CACHE PATH "TAPP TBLIS binary install directory")
set(TAPP_TBLIS_INSTALL_INCLUDEDIR "include"
        CACHE PATH "TAPP TBLIS INCLUDE install directory")
set(TAPP_TBLIS_INSTALL_LIBDIR "lib"
        CACHE PATH "TAPP TBLIS LIB install directory")
set(TAPP_TBLIS_INSTALL_DATADIR "share/mpqc/${TAPP_TBLIS_EXT_VERSION}/data"
        CACHE PATH "TAPP TBLIS DATA install directory")
set(TAPP_TBLIS_INSTALL_DOCDIR "share/tapp/${TAPP_TBLIS_EXT_VERSION}/doc"
        CACHE PATH "TAPP TBLIS DOC install directory")
set(TAPP_TBLIS_INSTALL_CMAKEDIR "lib/cmake/mpqc"
        CACHE PATH "TAPP TBLIS CMAKE install directory")

# Expects the tapp-api target and the tblis-static target (FetchContent in the
# top-level CMakeLists.txt) to be available.

find_package(Threads REQUIRED)

add_library(tapp-tblis SHARED)

target_sources(
  tapp-tblis
  PRIVATE
    src/error.cpp
    src/executor.cpp
    src/runtime.cpp
    src/tensor.cpp
    src/product.cpp
    src/tblis_utils.cpp
    include/datatype.h
    include/error.h
    include/executor.h
    include/tensor.h
    include/product.h
    include/tblis_utils.h
  )

set_property(
   TARGET tapp-tblis
   PROPERTY
     CXX_STANDARD 20
     CXX_STANDARD_REQUIRED YES
     CXX_EXTENSIONS NO
)

target_include_directories(
  tapp-tblis
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  tapp-tblis
  PUBLIC
    tapp-api
  PRIVATE
    tblis-static
    Threads::Threads
)
