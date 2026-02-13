# cuTENSOR is not part of the CUDA toolkit; look for it separately
if(NOT TARGET cutensor::cutensor)
  find_path(CUTENSOR_INCLUDE_DIR
    NAMES cutensor.h
    HINTS ${CUTENSOR_ROOT} ENV CUTENSOR_ROOT
          ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES include
  )
  find_library(CUTENSOR_LIBRARY
    NAMES cutensor
    HINTS ${CUTENSOR_ROOT} ENV CUTENSOR_ROOT
          ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES lib lib64 lib/${CMAKE_LIBRARY_ARCHITECTURE}
  )

  if(NOT CUTENSOR_INCLUDE_DIR OR NOT CUTENSOR_LIBRARY)
    message(FATAL_ERROR "cuTENSOR not found; set CUTENSOR_ROOT to the cuTENSOR installation prefix")
  endif()
  message(STATUS "Found cuTENSOR: ${CUTENSOR_LIBRARY}")
  message(STATUS "cuTENSOR include: ${CUTENSOR_INCLUDE_DIR}")

  add_library(cutensor::cutensor UNKNOWN IMPORTED GLOBAL)
  set_target_properties(cutensor::cutensor PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_DIR}"
  )
endif()

add_library(tapp-cutensor SHARED)
set_property(TARGET tapp-cutensor PROPERTY EXPORT_NAME cutensor)
add_library(tapp::cutensor ALIAS tapp-cutensor)
target_link_libraries(
        cutensor::cutensor
        INTERFACE
        CUDA::cudart
)

target_sources(tapp-cutensor
  PRIVATE
    src/attributes.cpp
    src/datatype.cpp
    src/error.cpp
    src/executor.cpp
    src/handle.cpp
    src/product.cpp
    src/tensor.cpp
)

set_target_properties(tapp-cutensor PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

target_include_directories(tapp-cutensor
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tapp-cutensor
  PUBLIC
    tapp-api
  PRIVATE
    cutensor::cutensor
    CUDA::cudart
)

install(TARGETS tapp-cutensor EXPORT tapp
  COMPONENT cutensor)

if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  target_link_options(tapp-cutensor PRIVATE "-undefined;dynamic_lookup")
endif()
