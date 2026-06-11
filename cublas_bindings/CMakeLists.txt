# cuBLAS/TTGT bindings for TAPP.
#
# Implements the TAPP tensor-product API directly in src/ using the TTGT
# (transpose-transpose-GEMM-transpose) scheme: cuBLAS GEMM plus cuTT transposes.
# cuBLAS ships with the CUDA toolkit; cuTT does not and is located via CUTT_ROOT.

# EMULATION enables the cuBLAS fixed-point FP64 emulation path (variable mantissa
# size, driven by the requested precision digits). Off by default; requires
# CUDA >= 13.
if(NOT DEFINED TAPP_CUBLAS_EMULATION)
  set(TAPP_CUBLAS_EMULATION 0)
endif()

# cuTT (transpose library) - not part of the CUDA toolkit.
if(NOT TARGET cutt)
  if(NOT DEFINED CUTT_ROOT AND DEFINED ENV{CUTT_ROOT})
    set(CUTT_ROOT $ENV{CUTT_ROOT})
  endif()
  if(NOT CUTT_ROOT)
    message(FATAL_ERROR "cuTT not found; set CUTT_ROOT to the cuTT installation prefix")
  endif()
  message(STATUS "Using cuTT from: ${CUTT_ROOT}")
  add_library(cutt STATIC IMPORTED GLOBAL)
  set_target_properties(cutt PROPERTIES
    IMPORTED_LOCATION "${CUTT_ROOT}/lib/libcutt.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTT_ROOT}/include"
  )
endif()

add_library(tapp-cublas SHARED)
set_property(TARGET tapp-cublas PROPERTY EXPORT_NAME cublas)
add_library(tapp::cublas ALIAS tapp-cublas)

target_sources(tapp-cublas
  PRIVATE
    src/product.cu
    src/attributes.cpp
    src/datatype.cpp
    src/error.cpp
    src/executor.cpp
    src/handle.cpp
    src/status.cpp
    src/tensor.cpp
)

set_target_properties(tapp-cublas PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES
)

# Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> to target a specific GPU (e.g. 80 for
# A100, 90 for H100/Grace-Hopper). When unset, fall back to a portable list
# rather than "native" so the library can be built on a GPU-less login node.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(tapp-cublas PROPERTIES CUDA_ARCHITECTURES "70;80;90")
endif()

target_compile_definitions(tapp-cublas PRIVATE EMULATION=${TAPP_CUBLAS_EMULATION})

target_include_directories(tapp-cublas
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tapp-cublas
  PUBLIC
    tapp::api
  PRIVATE
    CUDA::cudart
    CUDA::cublas
    cutt
)

install(TARGETS tapp-cublas EXPORT tapp
  COMPONENT cublas)
