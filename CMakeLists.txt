cmake_minimum_required(VERSION 3.15)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "Install Prefix" FORCE)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output" FORCE)
project(tapp LANGUAGES C CXX)

add_library(tapp SHARED)

target_sources(
  tapp
  PUBLIC
    src/tapp.h
    src/tapp/tapp_ex_imp.h
  PRIVATE
    src/tapp/tensor.h
    src/tapp/product.h
    src/tapp/attributes.h
    src/tapp/datatype.h
    src/tapp/error.c
    src/tapp/error.h
    src/tapp/executor.c
    src/tapp/executor.h
    src/tapp/handle.c
    src/tapp/handle.h
    src/tapp/status.h
    src/tapp/tensor.c
    src/tapp/product.c
  )

set_property(
   TARGET tapp
   PROPERTY
     CXX_STANDARD 20
     CXX_STANDARD_REQUIRED YES
     CXX_EXTENSIONS NO
)

target_include_directories(
  tapp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tapp
  )


  enable_testing()

  option(ENABLE_TBLIS "Build and link TBLIS and TBLIS bindings" OFF)

if(ENABLE_TBLIS)

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

  target_compile_definitions(tapp PRIVATE ENABLE_TBLIS=1)

  target_sources(
  tapp
  PRIVATE
    tblis_bindings/tblis_bind.h
    tblis_bindings/tblis_bind.cpp
  )
  target_include_directories(
  tapp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tblis_bindings
  )

  # ----------------------------------------------------------------------------
  # interface

  target_link_directories( # can be removed?
    tapp
    PUBLIC
      ${TBLIS_INSTALL_DIR}/lib
  )

  target_link_libraries(
    tapp
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
      tapp
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
     tapp # Linking to tapp provides everything needed.
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