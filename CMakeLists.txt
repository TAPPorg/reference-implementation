cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

# see https://semver.org/
set (TAPP_REFERENCE_MAJOR_VERSION 0)
set (TAPP_REFERENCE_MINOR_VERSION 5)
set (TAPP_REFERENCE_PATCH_VERSION 0)
set (TAPP_REFERENCE_PRERELEASE_ID )
set (TAPP_REFERENCE_BUILD_ID )

set(TAPP_REFERENCE_VERSION "${TAPP_REFERENCE_MAJOR_VERSION}.${TAPP_REFERENCE_MINOR_VERSION}.${TAPP_REFERENCE_PATCH_VERSION}")
if (TAPP_REFERENCE_PRERELEASE_ID)
  set(TAPP_REFERENCE_EXT_VERSION "${TAPP_REFERENCE_VERSION}-${TAPP_REFERENCE_PRERELEASE_ID}")
else(TAPP_REFERENCE_PRERELEASE_ID)
  set(TAPP_REFERENCE_EXT_VERSION "${TAPP_REFERENCE_VERSION}")
endif(TAPP_REFERENCE_PRERELEASE_ID)
if (TAPP_REFERENCE_BUILD_ID)
  set(TAPP_REFERENCE_EXT_VERSION "${TAPP_REFERENCE_EXT_VERSION}+${TAPP_REFERENCE_BUILD_ID}")
endif(TAPP_REFERENCE_BUILD_ID)

# Extract the git revision tag information
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
          COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_VARIABLE TAPP_REFERENCE_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
          TAPP_REFERENCE_REVISION "${TAPP_REFERENCE_REVISION}")
else()
  set(TAPP_REFERENCE_REVISION "unknown")
endif()

project(tapp-reference
        VERSION ${TAPP_REFERENCE_VERSION}
        DESCRIPTION "Reference Implementation of TAPP (Tensor Algebra Processing Primitives)"
        LANGUAGES C
        HOMEPAGE_URL "https://github.com/TAPPOrg/")

include(GNUInstallDirs)

set(TAPP_REFERENCE_INSTALL_BINDIR "bin"
        CACHE PATH "TAPP REFERENCE binary install directory")
set(TAPP_REFERENCE_INSTALL_INCLUDEDIR "include"
        CACHE PATH "TAPP REFERENCE INCLUDE install directory")
set(TAPP_REFERENCE_INSTALL_LIBDIR "lib"
        CACHE PATH "TAPP REFERENCE LIB install directory")
set(TAPP_REFERENCE_INSTALL_DATADIR "share/mpqc/${TAPP_REFERENCE_EXT_VERSION}/data"
        CACHE PATH "TAPP REFERENCE DATA install directory")
set(TAPP_REFERENCE_INSTALL_DOCDIR "share/tapp/${TAPP_REFERENCE_EXT_VERSION}/doc"
        CACHE PATH "TAPP REFERENCE DOC install directory")
set(TAPP_REFERENCE_INSTALL_CMAKEDIR "lib/cmake/mpqc"
        CACHE PATH "TAPP REFERENCE CMAKE install directory")

# this provides tapp-api target
add_subdirectory(api)

# this provides multi-tapp target
add_subdirectory(multi-tapp)

add_library(tapp-reference SHARED)

target_sources(
  tapp-reference
  PRIVATE
    reference_implementation/include/ref_impl.h
    reference_implementation/src/error.c
    reference_implementation/src/executor.c
    reference_implementation/src/handle.c
    reference_implementation/src/tensor.c
    reference_implementation/src/product.c
  )

set_property(
   TARGET tapp-reference
   PROPERTY
     C_STANDARD 99
     C_STANDARD_REQUIRED YES
     C_EXTENSIONS NO
)

target_include_directories(
  tapp-reference
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/reference_implementation/include
)
if (APPLE AND CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  target_link_options(tapp-reference PRIVATE "-undefined;dynamic_lookup")
endif()

target_link_libraries(tapp-reference PUBLIC tapp-api)

enable_testing()

option(TAPP_REFERENCE_ENABLE_TBLIS "Build and link TBLIS and TBLIS bindings" OFF)
option(TAPP_REFERENCE_BUILD_EXERCISE "Build contraction exercise with TODOs in it." OFF)

option(TAPP_REFERENCE_ENABLE_F16 "Turn on F16 support" OFF)
option(TAPP_REFERENCE_ENABLE_BF16 "Turn on BF16 support" OFF)

if(TAPP_REFERENCE_ENABLE_F16)
  target_compile_definitions(tapp-reference PRIVATE TAPP_REFERENCE_ENABLE_F16=1)
endif()

if(TAPP_REFERENCE_ENABLE_BF16)
  target_compile_definitions(tapp-reference PRIVATE TAPP_REFERENCE_ENABLE_BF16=1)
endif()

if(TAPP_REFERENCE_ENABLE_TBLIS)

  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
  else()
    message(FATAL_ERROR "Cannot build TBLIS as part of TAPP due to missing CXX language support; ensure that the CXX compiler can be discovered")
  endif()

  set(TBLIS_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/tblis)

  include(FetchContent)

  FetchContent_Declare(
    tblis
    GIT_REPOSITORY https://github.com/devinamatthews/tblis.git
    GIT_TAG 9b95712
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/tblis
    UPDATE_DISCONNECTED TRUE
  )

  FetchContent_MakeAvailable(tblis)

  target_compile_definitions(tapp-reference PRIVATE TAPP_REFERENCE_ENABLE_TBLIS=1)

  target_sources(
  tapp-reference
  PRIVATE
    tblis_bindings/tblis_bind.h
    tblis_bindings/tblis_bind.cpp
  )
  set_property(
          TARGET tapp-reference
          PROPERTY
          CXX_STANDARD 20
          CXX_STANDARD_REQUIRED YES
          CXX_EXTENSIONS NO
  )
  target_include_directories(
  tapp-reference
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tblis_bindings
  )

  # ----------------------------------------------------------------------------
  # interface

  target_link_libraries(
    tapp-reference
    PUBLIC
      tblis-static
  )

  # ----------------------------------------------------------------------------
  # testing

  add_executable(test++)

  target_sources(
    test++
    PRIVATE
      test/test.cpp
      test/test.h
    )

  target_link_libraries(
    test++
    PRIVATE
      tapp-reference
    )

  set_property(
    TARGET test++
    PROPERTY
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
  )

  add_test(
    NAME test++
    COMMAND $<TARGET_FILE:test++>
    )

endif()

add_executable(demo)

target_sources(
  demo
  PRIVATE
    test/demo.c
    test/helpers.c
    test/helpers.h
  )

  target_link_libraries(
   demo
   PRIVATE
     tapp-reference
  )


add_test(
  NAME demo
  COMMAND $<TARGET_FILE:demo>
  )

add_executable(driver)

target_sources(
  driver
  PRIVATE
    examples/driver/driver.c
    test/helpers.c
    test/helpers.h
  )

  target_include_directories(
  driver
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )

  target_link_libraries(
   driver
   PRIVATE
     tapp-reference
  )


add_test(
  NAME driver
  COMMAND $<TARGET_FILE:driver>
  )

add_executable(multi-tapp-driver)

target_sources(
  multi-tapp-driver
  PRIVATE
    test/multi-tapp-driver.c
    test/helpers.c
    test/helpers.h
  )

  target_include_directories(
  driver
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )

  target_link_libraries(
   multi-tapp-driver
   PRIVATE
     multi-tapp
  )


add_test(
  NAME multi-tapp-driver
  COMMAND $<TARGET_FILE:driver>
  )

if(TAPP_REFERENCE_BUILD_EXERCISE)
  add_executable(exercise_contraction)
  
  target_sources(
    exercise_contraction
    PRIVATE
      examples/exercise_contraction/exercise_contraction.c
      test/helpers.c
      test/helpers.h
    )
  
    target_include_directories(
    exercise_contraction
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )

  target_link_libraries(
   exercise_contraction
   PRIVATE
     tapp-reference
  )


  add_test(
    NAME exercise_contraction
    COMMAND $<TARGET_FILE:exercise_contraction>
    )
endif()

add_executable(exercise_contraction_answers)

target_sources(
  exercise_contraction_answers
  PRIVATE
    examples/exercise_contraction/answers/exercise_contraction_answers.c
    test/helpers.c
    test/helpers.h
  )

  target_include_directories(
  exercise_contraction_answers
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )

  target_link_libraries(
   exercise_contraction_answers
   PRIVATE
     tapp-reference
  )


add_test(
  NAME exercise_contraction_answers
  COMMAND $<TARGET_FILE:exercise_contraction_answers>
  )

  add_library(exercise_tucker SHARED)

target_sources(
  exercise_tucker
  PUBLIC
    examples/exercise_tucker/tapp_tucker/exercise_tucker.h
  PRIVATE
    examples/exercise_tucker/tapp_tucker/exercise_tucker.c
  )

  target_link_libraries(
   exercise_tucker
   PRIVATE
     tapp-reference
  )

  add_library(exercise_tucker_answers SHARED)

target_sources(
  exercise_tucker_answers
  PUBLIC
    examples/exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.h
  PRIVATE
    examples/exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.c
  )

  target_link_libraries(
   exercise_tucker_answers
   PRIVATE
     tapp-reference
  )
