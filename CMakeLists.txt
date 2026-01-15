cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output")

project(tapp-reference LANGUAGES C)

# this provides TAPP API target
add_subdirectory(api)

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

option(ENABLE_TBLIS "Build and link TBLIS and TBLIS bindings" OFF)
option(BUILD_EXERCISE "Build contraction exercise with TODOs in it." OFF)

option(ENABLE_F16 "Turn on F16 support" OFF)
option(ENABLE_BF16 "Turn on BF16 support" OFF)

if(ENABLE_F16)
  target_compile_definitions(tapp-reference PRIVATE ENABLE_F16=1)
endif()

if(ENABLE_BF16)
  target_compile_definitions(tapp-reference PRIVATE ENABLE_BF16=1)
endif()

if(ENABLE_TBLIS)

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

  target_compile_definitions(tapp-reference PRIVATE ENABLE_TBLIS=1)

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

  target_link_directories( # can be removed?
    tapp-reference
    PUBLIC
      ${TBLIS_INSTALL_DIR}/lib
  )

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
     tapp-reference # Linking to tapp provides everything needed.
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
     tapp # Linking to tapp provides everything needed.
  )


add_test(
  NAME driver
  COMMAND $<TARGET_FILE:driver>
  )

if(BUILD_EXERCISE)
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
     tapp # Linking to tapp provides everything needed.
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
     tapp # Linking to tapp provides everything needed.
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
     tapp # Linking to tapp provides everything needed.
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
     tapp # Linking to tapp provides everything needed.
  )
