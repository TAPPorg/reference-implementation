add_library(tapp-reference SHARED)
set_property(TARGET tapp-reference PROPERTY EXPORT_NAME reference)
add_library(tapp::reference ALIAS tapp-reference)

target_sources(
  tapp-reference
  PRIVATE
    include/ref_impl.h
    src/error.c
    src/executor.c
    src/handle.c
    src/tensor.c
    src/product.c
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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${TAPP_INSTALL_INCLUDEDIR}>
)
if (APPLE AND CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  target_link_options(tapp-reference PRIVATE "-undefined;dynamic_lookup")
endif()

target_link_libraries(tapp-reference PUBLIC tapp-api)

option(TAPP_BUILD_EXERCISE "Build contraction exercise with TODOs in it." OFF)

option(TAPP_REFERENCE_ENABLE_F16 "Turn on F16 support" OFF)
option(TAPP_REFERENCE_ENABLE_BF16 "Turn on BF16 support" OFF)

if(TAPP_REFERENCE_ENABLE_F16)
  target_compile_definitions(tapp-reference PRIVATE TAPP_REFERENCE_ENABLE_F16=1)
endif()

if(TAPP_REFERENCE_ENABLE_BF16)
  target_compile_definitions(tapp-reference PRIVATE TAPP_REFERENCE_ENABLE_BF16=1)
endif()

if(TAPP_REFERENCE_USE_TBLIS)

  set(TBLIS_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/tblis)

  include(FetchContent)

  FetchContent_Declare(
    tblis
    GIT_REPOSITORY https://github.com/devinamatthews/tblis.git
    GIT_TAG 9b95712
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/tblis
    UPDATE_DISCONNECTED TRUE
    EXCLUDE_FROM_ALL
  )

  FetchContent_MakeAvailable(tblis)

  target_compile_definitions(tapp-reference PRIVATE TAPP_REFERENCE_USE_TBLIS=1)

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
    PRIVATE
      tblis-static
  )

endif()

install(TARGETS tapp-reference EXPORT tapp
        COMPONENT reference
        LIBRARY DESTINATION "${TAPP_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${TAPP_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${TAPP_INSTALL_BINDIR}")
# N.B. no headers need to be installed! Can use API headers directly
