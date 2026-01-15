cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

# see https://semver.org/
set (TAPP_API_MAJOR_VERSION 0)
set (TAPP_API_MINOR_VERSION 5)
set (TAPP_API_PATCH_VERSION 0)
set (TAPP_API_PRERELEASE_ID )
set (TAPP_API_BUILD_ID )

set(TAPP_API_VERSION "${TAPP_API_MAJOR_VERSION}.${TAPP_API_MINOR_VERSION}.${TAPP_API_PATCH_VERSION}")
if (TAPP_API_PRERELEASE_ID)
  set(TAPP_API_EXT_VERSION "${TAPP_API_VERSION}-${TAPP_API_PRERELEASE_ID}")
else(TAPP_API_PRERELEASE_ID)
  set(TAPP_API_EXT_VERSION "${TAPP_API_VERSION}")
endif(TAPP_API_PRERELEASE_ID)
if (TAPP_API_BUILD_ID)
  set(TAPP_API_EXT_VERSION "${TAPP_API_EXT_VERSION}+${TAPP_API_BUILD_ID}")
endif(TAPP_API_BUILD_ID)

# Extract the git revision tag information
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
          COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_VARIABLE TAPP_API_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
          TAPP_API_REVISION "${TAPP_API_REVISION}")
else()
  set(TAPP_API_REVISION "unknown")
endif()

project(tapp-api
        VERSION ${TAPP_API_VERSION}
        DESCRIPTION "TAPP: Tensor Algebra Processing Primitives"
        LANGUAGES C
        HOMEPAGE_URL "https://github.com/TAPPOrg/")

include(GNUInstallDirs)

set(TAPP_API_INSTALL_BINDIR "bin"
        CACHE PATH "TAPP API binary install directory")
set(TAPP_API_INSTALL_INCLUDEDIR "include"
        CACHE PATH "TAPP API INCLUDE install directory")
set(TAPP_API_INSTALL_LIBDIR "lib"
        CACHE PATH "TAPP API LIB install directory")
set(TAPP_API_INSTALL_DATADIR "share/mpqc/${TAPP_API_EXT_VERSION}/data"
        CACHE PATH "TAPP API DATA install directory")
set(TAPP_API_INSTALL_DOCDIR "share/tapp/${TAPP_API_EXT_VERSION}/doc"
        CACHE PATH "TAPP API DOC install directory")
set(TAPP_API_INSTALL_CMAKEDIR "lib/cmake/mpqc"
        CACHE PATH "TAPP API CMAKE install directory")

add_library(tapp-api INTERFACE)

target_sources(
  tapp-api
  PUBLIC
    include/tapp.h
  PRIVATE
    include/tapp/tensor.h
    include/tapp/product.h
    include/tapp/attributes.h
    include/tapp/datatype.h
    include/tapp/error.h
    include/tapp/executor.h
    include/tapp/handle.h
    include/tapp/status.h
  )

set_property(
   TARGET tapp-api
   PROPERTY
     C_STANDARD 99
     C_STANDARD_REQUIRED YES
     C_EXTENSIONS NO
)

target_include_directories(tapp-api INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${TAPP_INTERFACE_INSTALL_INCLUDEDIR}>
)
