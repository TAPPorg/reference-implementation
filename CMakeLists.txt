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

  include(ExternalProject)

  ExternalProject_Add(
    tblis
    GIT_REPOSITORY https://github.com/devinamatthews/tblis.git
    GIT_TAG 4de1919
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/tblis
    CONFIGURE_COMMAND <SOURCE_DIR>/configure
                      --prefix=${TBLIS_INSTALL_DIR}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    UPDATE_DISCONNECTED TRUE
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    CMAKE_ARGS
            -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
  )

  add_library(tblis-interface INTERFACE)
  add_dependencies(tblis-interface tblis)
  target_include_directories(tblis-interface INTERFACE
      "${TBLIS_INSTALL_DIR}/include"
      "${TBLIS_INSTALL_DIR}/include/tblis"
  )
  set(TBLIS_LIBRARY_PATH "")
  if(WIN32)
      set(TBLIS_LIBRARY_PATH "${TBLIS_INSTALL_DIR}/lib/tblis.lib")
  else()
      set(TBLIS_LIBRARY_PATH "${TBLIS_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}tblis${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()
  
  target_link_libraries(tblis-interface INTERFACE
      "${TBLIS_LIBRARY_PATH}"
  )

  # Ensure `tblis` is built before the demo executable.
  target_compile_definitions(tapp PRIVATE ENABLE_TBLIS=1)
  add_dependencies(tapp tblis)

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
      tblis-interface
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
