cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

# see https://semver.org/
set (TAPP_MAJOR_VERSION 0)
set (TAPP_MINOR_VERSION 5)
set (TAPP_PATCH_VERSION 0)
set (TAPP_PRERELEASE_ID )
set (TAPP_BUILD_ID )

set(TAPP_VERSION "${TAPP_MAJOR_VERSION}.${TAPP_MINOR_VERSION}.${TAPP_PATCH_VERSION}")
if (TAPP_PRERELEASE_ID)
  set(TAPP_EXT_VERSION "${TAPP_VERSION}-${TAPP_PRERELEASE_ID}")
else(TAPP_PRERELEASE_ID)
  set(TAPP_EXT_VERSION "${TAPP_VERSION}")
endif(TAPP_PRERELEASE_ID)
if (TAPP_BUILD_ID)
  set(TAPP_EXT_VERSION "${TAPP_EXT_VERSION}+${TAPP_BUILD_ID}")
endif(TAPP_BUILD_ID)

# Extract the git revision tag information
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
          COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_VARIABLE TAPP_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
          TAPP_REVISION "${TAPP_REVISION}")
else()
  set(TAPP_REVISION "unknown")
endif()

project(tapp
        VERSION ${TAPP_VERSION}
        DESCRIPTION "TAPP (Tensor Algebra Processing Primitives)"
        LANGUAGES C
        HOMEPAGE_URL "https://github.com/TAPPOrg/")

# TBLIS requires CXX; enable_language must be called at the top level
option(TAPP_REFERENCE_USE_TBLIS "TAPP-Reference will use TBLIS to implement TAPP_product" OFF)
if(TAPP_REFERENCE_USE_TBLIS)
  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
  else()
    message(FATAL_ERROR "Cannot build TBLIS as part of TAPP due to missing CXX language support; ensure that the CXX compiler can be discovered")
  endif()
endif()

include(GNUInstallDirs)

set(TAPP_INSTALL_BINDIR "bin"
        CACHE PATH "TAPP binary install directory")
set(TAPP_INSTALL_INCLUDEDIR "include"
        CACHE PATH "TAPP include install directory")
set(TAPP_INSTALL_LIBDIR "lib"
        CACHE PATH "TAPP library install directory")
set(TAPP_INSTALL_CMAKEDIR "lib/cmake/tapp"
        CACHE PATH "TAPP CMake config install directory")
set(TAPP_INSTALL_DATADIR "share/tapp/${TAPP_EXT_VERSION}/data"
        CACHE PATH "TAPP data install directory")
set(TAPP_INSTALL_DOCDIR "share/tapp/${TAPP_EXT_VERSION}/doc"
        CACHE PATH "TAPP doc install directory")

# this provides tapp::api target
add_subdirectory(api)

# this provides tapp::reference target
add_subdirectory(reference_implementation)

# ----------------------------------------------------------------------------
# cutensor bindings
option(TAPP_CUTENSOR "Build cuTensor bindings" OFF)
if (TAPP_CUTENSOR)
  # enable_language must be called at the top level
  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
  else()
    message(FATAL_ERROR "Cannot build cuTENSOR bindings due to missing CXX language support")
  endif()
  # since CUDAToolkit will be needed in tests/ also, load it here
  cmake_minimum_required(VERSION 3.17) # CUDAToolkit
  find_package(CUDAToolkit REQUIRED)

  add_subdirectory(cutensor_bindings)
endif()

# ----------------------------------------------------------------------------
# testing

include(CTest)

if(BUILD_TESTING)
  add_subdirectory(test)
  add_subdirectory(examples)
endif()

# ============================================================================
# CMake package config / export
# ============================================================================
include(CMakePackageConfigHelpers)

# version file
write_basic_package_version_file(tapp-config-version.cmake
    VERSION ${TAPP_VERSION} COMPATIBILITY AnyNewerVersion)

# export targets for the build tree
export(EXPORT tapp
       NAMESPACE tapp::
       FILE "${PROJECT_BINARY_DIR}/tapp-targets.cmake")

# install target export file
install(EXPORT tapp
    FILE "tapp-targets.cmake"
    DESTINATION "${TAPP_INSTALL_CMAKEDIR}"
    NAMESPACE tapp::)

# configure the config file
configure_package_config_file(cmake/tapp-config.cmake.in
    "${PROJECT_BINARY_DIR}/tapp-config.cmake"
    INSTALL_DESTINATION "${TAPP_INSTALL_CMAKEDIR}")

install(FILES
    "${PROJECT_BINARY_DIR}/tapp-config.cmake"
    "${PROJECT_BINARY_DIR}/tapp-config-version.cmake"
    DESTINATION "${TAPP_INSTALL_CMAKEDIR}")
