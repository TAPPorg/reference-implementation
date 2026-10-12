# ----------------------------------------------------------------------------
# tapp-tblis: TAPP implementation backed by TBLIS
#
# Expects the tapp-api target and the tblis-static target (FetchContent in the
# top-level CMakeLists.txt) to be available.

find_package(Threads REQUIRED)

add_library(tapp-tblis SHARED)

target_sources(
  tapp-tblis
  PRIVATE
    src/error.cpp
    src/executor.cpp
    src/tensor.cpp
    src/product.cpp
    src/tblis_utils.cpp
    include/datatype.h
    include/error.h
    include/executor.h
    include/tensor.h
    include/product.h
    include/tblis_utils.h
  )

set_property(
  TARGET tapp-tblis
  PROPERTY
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(
  tapp-tblis
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  tapp-tblis
  PUBLIC
    tapp-api
  PRIVATE
    tblis-static
    Threads::Threads
)
