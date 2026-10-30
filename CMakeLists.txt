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
set_target_properties(tapp-reference PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/reference_implementation"
)

target_sources(
  tapp-reference
  PRIVATE
    reference_implementation/src/error.c
    reference_implementation/src/executor.c
    reference_implementation/src/runtime.c
    reference_implementation/src/tensor.c
    reference_implementation/src/product.c
    reference_implementation/src/product_template.c
    reference_implementation/src/product_instantiations.gen.c
    reference_implementation/src/generate_product_instantiations.py
    reference_implementation/src/internal.h
  )

set_source_files_properties(
  reference_implementation/src/product_template.c
  reference_implementation/src/product_instantiations.gen.c
  reference_implementation/src/generate_product_instantiations.py
  PROPERTIES HEADER_FILE_ONLY ON
  )

set_property(
   TARGET tapp-reference
   PROPERTY
     C_STANDARD 99
     C_STANDARD_REQUIRED YES
     C_EXTENSIONS NO
)

if (APPLE AND CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  target_link_options(tapp-reference PRIVATE "-undefined;dynamic_lookup")
endif()

target_link_libraries(tapp-reference PUBLIC tapp-api)

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

  # TCI (the threading layer of TBLIS) passes OpenMP_C_FLAGS as one quoted argument, which breaks with AppleClang
  # ("-Xclang -fopenmp") and leaves out the include directory of omp.h. Use pthreads for TCI on Apple instead
  if(APPLE)
    set(ENABLE_THREAD_MODEL pthreads CACHE STRING "TCI thread model")
  endif()

  FetchContent_MakeAvailable(tblis)

  # this provides tapp-tblis target
  add_subdirectory(tblis)

endif()

if(TAPP_REFERENCE_ENABLE_CUTENSOR)

  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
  else()
    message(FATAL_ERROR "Cannot build TAPP with cuTENSOR due to missing CXX language support; ensure that the CXX compiler can be discovered")
  endif()

  target_compile_definitions(tapp-reference PRIVATE TAPP_REFERENCE_ENABLE_CUTENSOR=1)
  find_package(CUDAToolkit REQUIRED)

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

  # this provides tapp-cutensor target
  add_subdirectory(cutensor)

endif()

option(TAPP_REFERENCE_ENABLE_CUBLAS "Build cuBLAS/TTGT bindings" OFF)

if(TAPP_REFERENCE_ENABLE_CUBLAS)

  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
  else()
    message(FATAL_ERROR "Cannot build TAPP with cuBLAS due to missing CXX language support; ensure that the CXX compiler can be discovered")
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "Cannot build TAPP with cuBLAS due to missing CUDA language support; ensure that the CUDA compiler can be discovered")
  endif()

  # CUDAToolkit provides CUDA::cudart and CUDA::cublas
  find_package(CUDAToolkit REQUIRED)

  set(TAPP_OZAKI_EMULATION_VERSION 0 CACHE STRING
      "FP64 emulation in the cuBLAS bindings: 0 = none, 1 = cuBLAS fixed-point emulation (CUDA >= 13), 2 = Ozaki-II via GEMMul8")
  set_property(CACHE TAPP_OZAKI_EMULATION_VERSION PROPERTY STRINGS 0 1 2)

  # cuTT (transpose library) is not part of the CUDA toolkit. Use the installation
  # at CUTT_ROOT if given, otherwise fetch and build it
  if(NOT DEFINED CUTT_ROOT AND DEFINED ENV{CUTT_ROOT})
    set(CUTT_ROOT $ENV{CUTT_ROOT})
  endif()
  if(CUTT_ROOT)
    message(STATUS "Using cuTT from: ${CUTT_ROOT}")
    add_library(cutt STATIC IMPORTED GLOBAL)
    set_target_properties(cutt PROPERTIES
      IMPORTED_LOCATION "${CUTT_ROOT}/lib/libcutt.a"
      INTERFACE_INCLUDE_DIRECTORIES "${CUTT_ROOT}/include"
    )
  else()
    include(FetchContent)
    find_package(Git REQUIRED)

    # The ValeevGroup fork of cuTT, which adds a CMake build. It does not build with
    # CUDA 13 as is; cublas/cmake/cutt-cuda13.patch explains and fixes that
    FetchContent_Declare(
      cutt
      GIT_REPOSITORY https://github.com/ValeevGroup/cutt.git
      GIT_TAG 0e8685bf82910bc7435835f846e88f1b39f47f09
      PATCH_COMMAND ${GIT_EXECUTABLE} apply ${CMAKE_CURRENT_SOURCE_DIR}/cublas/cmake/cutt-cuda13.patch
      UPDATE_DISCONNECTED TRUE
    )

    FetchContent_MakeAvailable(cutt)

    # cuTT's target carries no include directory and is linked into the shared tapp-cublas
    target_include_directories(cutt INTERFACE ${cutt_SOURCE_DIR}/src)
    set_target_properties(cutt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set_target_properties(cutt PROPERTIES CUDA_ARCHITECTURES "80;90")
    endif()
  endif()

  # GEMMul8 (Ozaki-II), built in place at OZAKI_II_ROOT
  if(TAPP_OZAKI_EMULATION_VERSION STREQUAL "2")
    if(NOT DEFINED OZAKI_II_ROOT AND DEFINED ENV{OZAKI_II_ROOT})
      set(OZAKI_II_ROOT $ENV{OZAKI_II_ROOT})
    endif()
    if(NOT OZAKI_II_ROOT)
      message(FATAL_ERROR "OZAKI_II_ROOT not found; set OZAKI_II_ROOT to the GEMMul8 directory")
    endif()
    if(NOT OZAKI_II_COMPUTE_CAPABILITY)
      message(FATAL_ERROR "OZAKI_II_COMPUTE_CAPABILITY not set; set it to the GPU_ARCH GEMMul8 was built for (e.g. 90)")
    endif()
    message(STATUS "Using GEMMul8 from: ${OZAKI_II_ROOT}")
    add_library(gemmul8 STATIC IMPORTED GLOBAL)
    set_target_properties(gemmul8 PROPERTIES
      IMPORTED_LOCATION "${OZAKI_II_ROOT}/lib/libgemmul8.a"
      INTERFACE_INCLUDE_DIRECTORIES "${OZAKI_II_ROOT}/include"
    )
  endif()

  # this provides tapp-cublas target
  add_subdirectory(cublas)

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

    # Links only tapp-tblis: it exports the same TAPP_* symbols as tapp-reference
    # Linux only: it checks threads and CPU masks through /proc and the Linux affinity API
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      add_executable(tapp-tblis-test)

      target_sources(
        tapp-tblis-test
        PRIVATE
          test/tblis_test.cpp
        )

      target_link_libraries(
        tapp-tblis-test
        PRIVATE
          tapp-tblis
        )

      set_property(
        TARGET tapp-tblis-test
        PROPERTY
          CXX_STANDARD 20
          CXX_STANDARD_REQUIRED YES
          CXX_EXTENSIONS NO
      )

      add_test(
        NAME tapp-tblis-test
        COMMAND $<TARGET_FILE:tapp-tblis-test>
        )

      # The same test under OpenMP settings that change how many threads an executor gets and where they run
      foreach(omp_setting OMP_THREAD_LIMIT=2 OMP_DYNAMIC=true OMP_PROC_BIND=true)
        add_test(
          NAME tapp-tblis-test-${omp_setting}
          COMMAND $<TARGET_FILE:tapp-tblis-test>
          )
        set_tests_properties(
          tapp-tblis-test-${omp_setting}
          PROPERTIES
            ENVIRONMENT ${omp_setting}
          )
      endforeach()
    endif()
  endif()

  # ----------------------------------------------------------------------------
  # multi-tapp test

  add_executable(multi-tapp-test)

  target_sources(
    multi-tapp-test
    PRIVATE
      test/multi_tapp_test.c
    )

  target_link_libraries(
    multi-tapp-test
    PRIVATE
      multi-tapp
      m
    )

  # The implementations are loaded at runtime, so they are only passed as paths
  add_dependencies(multi-tapp-test tapp-reference)
  set(MULTI_TAPP_TEST_BACKENDS $<TARGET_FILE:tapp-reference>)
  if(TAPP_REFERENCE_ENABLE_TBLIS)
    add_dependencies(multi-tapp-test tapp-tblis)
    list(APPEND MULTI_TAPP_TEST_BACKENDS $<TARGET_FILE:tapp-tblis>)
  endif()

  add_test(
    NAME multi-tapp-test
    COMMAND $<TARGET_FILE:multi-tapp-test> ${MULTI_TAPP_TEST_BACKENDS}
    )

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

  # cuTENSOR demo
  if(TAPP_REFERENCE_ENABLE_CUTENSOR)
    add_executable(tapp-cutensor-demo)

    target_sources(
      tapp-cutensor-demo
      PRIVATE
        test/cutensor_demo.cpp
        test/helpers.c
        test/helpers.h
        cutensor/include/tapp_cutensor.h
      )

    target_include_directories(
      tapp-cutensor-demo
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cutensor/include
    )

    target_link_libraries(
      tapp-cutensor-demo
      PRIVATE
         tapp::cutensor
         CUDA::cudart
      )

    add_test(
      NAME tapp-cutensor-demo
      COMMAND $<TARGET_FILE:tapp-cutensor-demo>
      )
  endif()

  # cuBLAS demo
  if(TAPP_REFERENCE_ENABLE_CUBLAS)
    add_executable(tapp-cublas-demo)

    target_sources(
      tapp-cublas-demo
      PRIVATE
        test/cublas_demo.cpp
        cublas/include/tapp_cublas.h
      )

    target_include_directories(
      tapp-cublas-demo
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cublas/include
    )

    target_link_libraries(
      tapp-cublas-demo
      PRIVATE
         tapp::cublas
         CUDA::cudart
      )

    add_test(
      NAME tapp-cublas-demo
      COMMAND $<TARGET_FILE:tapp-cublas-demo>
      )
  endif()

  # ----------------------------------------------------------------------------
  # examples and exercises

  add_subdirectory(examples)

endif()
