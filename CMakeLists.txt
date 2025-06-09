cmake_minimum_required(VERSION 3.14)

project(tapp LANGUAGES C CXX)



add_library(tapp SHARED)

target_sources(
  tapp
  PUBLIC
    src/tapp/tensor.c
    src/tapp/tensor.h
    src/tapp/product.c
    src/tapp/product.h
  PRIVATE
    src/tapp.h
    src/tapp/attributes.h
    src/tapp/datatype.h
    src/tapp/error.c
    src/tapp/error.h
    src/tapp/executor.c
    src/tapp/executor.h
    src/tapp/handle.c
    src/tapp/handle.h
    src/tapp/status.h
    src/tapp/tapp_ex_imp.h
  )

target_include_directories(
  tapp
  PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/tapp
  )



option(ENABLE_TBLIS "Build and link TBLIS and TBLIS bindings" ON)

if(ENABLE_TBLIS)

  set(TBLIS_INSTALL_DIR ${CMAKE_BINARY_DIR}/tblis)

  include(ExternalProject)

  ExternalProject_Add(
    tblis
    GIT_REPOSITORY https://github.com/devinamatthews/tblis.git
    GIT_TAG 4de1919
    PREFIX ${CMAKE_BINARY_DIR}/_deps/tblis
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${TBLIS_INSTALL_DIR}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    UPDATE_DISCONNECTED TRUE
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
  )


  # ----------------------------------------------------------------------------
  # interface

  add_library(tblis_interface INTERFACE)

  target_sources(
    tblis_interface
    PRIVATE
      tblis_bindings/tblis_bind.h
      tblis_bindings/tblis_bind.cpp
    )

  target_include_directories(
    tblis_interface
    INTERFACE
      ${TBLIS_INSTALL_DIR}/include
      ${TBLIS_INSTALL_DIR}/include/tblis
  )

  target_link_directories(
    tblis_interface
    INTERFACE
      ${TBLIS_INSTALL_DIR}/lib
  )

  target_link_libraries(
    tblis_interface
    INTERFACE
      tblis
  )

  target_link_libraries(
    tapp
    PUBLIC
      tblis_interface
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

  enable_testing()

  add_test(
    NAME test++
    COMMAND $<TARGET_FILE:test++>
    )

endif()
