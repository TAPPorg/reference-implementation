cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

# see https://semver.org/
set (MULTI_TAPP_MAJOR_VERSION 0)
set (MULTI_TAPP_MINOR_VERSION 5)
set (MULTI_TAPP_PATCH_VERSION 0)
set (MULTI_TAPP_PRERELEASE_ID )
set (MULTI_TAPP_BUILD_ID )

set(MULTI_TAPP_VERSION "${MULTI_TAPP_MAJOR_VERSION}.${MULTI_TAPP_MINOR_VERSION}.${MULTI_TAPP_PATCH_VERSION}")
if (MULTI_TAPP_PRERELEASE_ID)
  set(MULTI_TAPP_EXT_VERSION "${MULTI_TAPP_VERSION}-${MULTI_TAPP_PRERELEASE_ID}")
else(MULTI_TAPP_PRERELEASE_ID)
  set(MULTI_TAPP_EXT_VERSION "${MULTI_TAPP_VERSION}")
endif(MULTI_TAPP_PRERELEASE_ID)
if (MULTI_TAPP_BUILD_ID)
  set(MULTI_TAPP_EXT_VERSION "${MULTI_TAPP_EXT_VERSION}+${MULTI_TAPP_BUILD_ID}")
endif(MULTI_TAPP_BUILD_ID)

# Extract the git revision tag information
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
          COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_VARIABLE MULTI_TAPP_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
          MULTI_TAPP_REVISION "${MULTI_TAPP_REVISION}")
else()
  set(MULTI_TAPP_REVISION "unknown")
endif()

project(tapp-api
        VERSION ${MULTI_TAPP_VERSION}
        DESCRIPTION "TAPP: Tensor Algebra Processing Primitives"
        LANGUAGES C
        HOMEPAGE_URL "https://github.com/TAPPOrg/")

include(GNUInstallDirs)

set(MULTI_TAPP_INSTALL_BINDIR "bin"
        CACHE PATH "TAPP API binary install directory")
set(MULTI_TAPP_INSTALL_INCLUDEDIR "include"
        CACHE PATH "TAPP API INCLUDE install directory")
set(MULTI_TAPP_INSTALL_LIBDIR "lib"
        CACHE PATH "TAPP API LIB install directory")
set(MULTI_TAPP_INSTALL_DATADIR "share/mpqc/${MULTI_TAPP_EXT_VERSION}/data"
        CACHE PATH "TAPP API DATA install directory")
set(MULTI_TAPP_INSTALL_DOCDIR "share/tapp/${MULTI_TAPP_EXT_VERSION}/doc"
        CACHE PATH "TAPP API DOC install directory")
set(MULTI_TAPP_INSTALL_CMAKEDIR "lib/cmake/mpqc"
        CACHE PATH "TAPP API CMAKE install directory")

add_library(multi-tapp SHARED)

target_sources(
  multi-tapp
  PRIVATE
    src/attributes.c
    src/error.c
    src/executor.c
    src/handle.c
    src/product.c
    src/status.c
    src/tensor.c
    include/attributes.h
    include/error.h
    include/executor.h
    include/handle.h
    include/product.h
    include/status.h
    include/tensor.h
  )

set_property(
   TARGET multi-tapp
   PROPERTY
     C_STANDARD 99
     C_STANDARD_REQUIRED YES
     C_EXTENSIONS NO
)

target_include_directories(
  multi-tapp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(multi-tapp PUBLIC tapp-api)