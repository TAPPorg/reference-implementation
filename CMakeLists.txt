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
option(TAPP_REFERENCE_ENABLE_TBLIS "Build and link TBLIS and TBLIS bindings" OFF)
if(TAPP_REFERENCE_ENABLE_TBLIS)
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

# this provides tapp-api target
add_subdirectory(api)

# this provides tapp-reference target
add_subdirectory(reference_implementation)

# ----------------------------------------------------------------------------
# cutensor bindings

if (TAPP_REFERENCE_BUILD_CUTENSOR_BINDINGS)
    add_subdirectory(cutensor_bindings)
endif()

# ----------------------------------------------------------------------------
# testing

include(CTest)

if(BUILD_TESTING)

  # ----------------------------------------------------------------------------
  # TBLIS test

  if(TAPP_REFERENCE_ENABLE_TBLIS)
    add_executable(tapp-reference-test++)

    target_sources(
      tapp-reference-test++
      PRIVATE
        test/test.cpp
        test/test.h
      )

    target_link_libraries(
      tapp-reference-test++
      PRIVATE
        tapp-reference
        tblis-static
      )

    set_property(
      TARGET tapp-reference-test++
      PROPERTY
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )

    add_test(
      NAME tapp-reference-test++
      COMMAND $<TARGET_FILE:tapp-reference-test++>
      )
  endif()

  # ----------------------------------------------------------------------------
  # demo

  add_executable(tapp-reference-demo)

  target_sources(
    tapp-reference-demo
    PRIVATE
      test/demo.c
      test/helpers.c
      test/helpers.h
    )

  target_link_libraries(
    tapp-reference-demo
    PRIVATE
      tapp-reference
    )

  add_test(
    NAME tapp-reference-demo
    COMMAND $<TARGET_FILE:tapp-reference-demo>
    )

  # ----------------------------------------------------------------------------
  # cutensor specific code

  if (TAPP_REFERENCE_BUILD_CUTENSOR_BINDINGS)
    # ----------------------------------------------------------------------------
    # cutensor demo

    include(CheckLanguage)
    check_language(CXX)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CXX)
      enable_language(CUDA)
    else()
      message(FATAL_ERROR "Cannot build cuTENSOR bindings as part of TAPP due to missing CUDA language support; ensure that the CUDA compiler can be discovered")
    endif()

    add_executable(tapp-reference-cutensor-demo)

    target_sources(
      tapp-reference-cutensor-demo
      PRIVATE
        test/cutensor_demo.cu
        test/helpers.c
        test/helpers.h
    )

    target_link_libraries(
      tapp-reference-cutensor-demo
      PRIVATE
        cutensor_bindings
    )

    target_include_directories(
      tapp-reference-cutensor-demo
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/test
      PRIVATE
        ${CUTENSOR_INCLUDE_DIR}
    )

    add_test(
      NAME tapp-reference-cutensor-demo
      COMMAND $<TARGET_FILE:tapp-reference-cutensor-demo>
    )

    # ----------------------------------------------------------------------------
    # demo using dynamic library

    add_executable(tapp-reference-demo-dynamic)

    target_sources(
      tapp-reference-demo-dynamic
      PRIVATE
        test/demo_dynamic.c
        test/helpers.c
        test/helpers.h
        api/include/tapp.h
    )

    target_include_directories(
      tapp-reference-demo-dynamic
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/api/include
      PRIVATE
        ${CUTENSOR_INCLUDE_DIR}
    )

    add_test(
      NAME tapp-reference-demo-dynamic
      COMMAND $<TARGET_FILE:tapp-reference-demo-dynamic>
    )

    target_link_libraries(
      tapp-reference-demo-dynamic
      PRIVATE
        ${CMAKE_DL_LIBS}
    )

    # ----------------------------------------------------------------------------
    # test using dynamic library

    add_executable(tapp-reference-test-dynamic)

    target_sources(
      tapp-reference-test-dynamic
      PRIVATE
        test/test_dynamic.cpp
        test/test_dynamic.h
        api/include/tapp.h
    )

    target_include_directories(
      tapp-reference-test-dynamic
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/api/include
      PRIVATE
        ${CUTENSOR_INCLUDE_DIR}
    )

    add_test(
      NAME tapp-reference-test-dynamic
      COMMAND $<TARGET_FILE:tapp-reference-test-dynamic>
    )

    target_link_libraries(
      tapp-reference-test-dynamic
      PRIVATE
        ${CMAKE_DL_LIBS}
    )

  endif()

  # ----------------------------------------------------------------------------
  # driver

  add_executable(tapp-reference-driver)

  target_sources(
    tapp-reference-driver
    PRIVATE
      examples/driver/driver.c
      test/helpers.c
      test/helpers.h
    )

  target_include_directories(
    tapp-reference-driver
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

  target_link_libraries(
    tapp-reference-driver
    PRIVATE
      tapp-reference
    )

  add_test(
    NAME tapp-reference-driver
    COMMAND $<TARGET_FILE:tapp-reference-driver>
    )

  # ----------------------------------------------------------------------------
  # exercise: contraction

  if(TAPP_BUILD_EXERCISE)
    add_executable(tapp-reference-exercise_contraction)

    target_sources(
      tapp-reference-exercise_contraction
      PRIVATE
        examples/exercise_contraction/exercise_contraction.c
        test/helpers.c
        test/helpers.h
      )

    target_include_directories(
      tapp-reference-exercise_contraction
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test
      )

    target_link_libraries(
      tapp-reference-exercise_contraction
      PRIVATE
        tapp-reference
      )

    add_test(
      NAME tapp-reference-exercise_contraction
      COMMAND $<TARGET_FILE:tapp-reference-exercise_contraction>
      )
  endif()

  # ----------------------------------------------------------------------------
  # exercise: contraction answers

  add_executable(tapp-reference-exercise_contraction_answers)

  target_sources(
    tapp-reference-exercise_contraction_answers
    PRIVATE
      examples/exercise_contraction/answers/exercise_contraction_answers.c
      test/helpers.c
      test/helpers.h
    )

  target_include_directories(
    tapp-reference-exercise_contraction_answers
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

  target_link_libraries(
    tapp-reference-exercise_contraction_answers
    PRIVATE
      tapp-reference
    )

  add_test(
    NAME tapp-reference-exercise_contraction_answers
    COMMAND $<TARGET_FILE:tapp-reference-exercise_contraction_answers>
    )

  # ----------------------------------------------------------------------------
  # exercise: tucker

  add_library(tapp-reference-exercise_tucker SHARED)

  target_sources(
    tapp-reference-exercise_tucker
    PUBLIC
      examples/exercise_tucker/tapp_tucker/exercise_tucker.h
    PRIVATE
      examples/exercise_tucker/tapp_tucker/exercise_tucker.c
    )

  target_link_libraries(
    tapp-reference-exercise_tucker
    PRIVATE
      tapp-reference
    )

  # ----------------------------------------------------------------------------
  # exercise: tucker answers

  add_library(tapp-reference-exercise_tucker_answers SHARED)

  target_sources(
    tapp-reference-exercise_tucker_answers
    PUBLIC
      examples/exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.h
    PRIVATE
      examples/exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.c
    )

  target_link_libraries(
    tapp-reference-exercise_tucker_answers
    PRIVATE
      tapp-reference
    )

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
