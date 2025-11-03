cmake_minimum_required(VERSION 3.15)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "Install Prefix" FORCE)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output" FORCE)
project(tapp LANGUAGES C CXX Fortran)

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
     C_STANDARD 99
     C_STANDARD_REQUIRED YES
     C_EXTENSIONS NO
)

target_include_directories(
  tapp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tapp
)
if (APPLE AND CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  target_link_options(tapp PRIVATE "-undefined;dynamic_lookup")
endif()

enable_testing()

option(ENABLE_TBLIS "Build and link TBLIS and TBLIS bindings" OFF)
option(BUILD_EXERCISE "Build contraction exercise with TODOs in it." OFF)
option(ENABLE_CTF "Link to manually installed Cyclops CTF library." ON)

option(ENABLE_F16 "Turn on F16 support" OFF)
option(ENABLE_BF16 "Turn on BF16 support" OFF)

if(ENABLE_F16)
  target_compile_definitions(tapp PRIVATE ENABLE_F16=1)
endif()

if(ENABLE_BF16)
  target_compile_definitions(tapp PRIVATE ENABLE_BF16=1)
endif()

if(ENABLE_CTF)
  message(STATUS "ENABLE_CTF is ON. Configuring CTF build.")

  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
  else()
    message(FATAL_ERROR "Cannot build CTF bindings due to missing CXX language support;
ensure that the CXX compiler can be discovered")
  endif()

  set(TAPP_MPI_PATH "" CACHE PATH "Root directory of the specific MPI installation to use")

  if(TAPP_MPI_PATH)
    # Set paths for all three MPI compiler wrappers
    set(MPI_C_COMPILER   "${TAPP_MPI_PATH}/bin/mpicc")
    set(MPI_CXX_COMPILER "${TAPP_MPI_PATH}/bin/mpicxx")

    if(EXISTS "${TAPP_MPI_PATH}/bin/mpif90")
      set(MPI_Fortran_COMPILER "${TAPP_MPI_PATH}/bin/mpif90")
    elseif(EXISTS "${TAPP_MPI_PATH}/bin/mpifc")
      set(MPI_Fortran_COMPILER "${TAPP_MPI_PATH}/bin/mpifc")
    endif()

    # Now that all compilers are set, FindMPI will work correctly
    if(EXISTS ${MPI_C_COMPILER} AND EXISTS ${MPI_CXX_COMPILER})
        find_package(MPI REQUIRED)
    else()
        message(FATAL_ERROR "TAPP: Could not find mpicc/mpicxx in ${TAPP_MPI_PATH}/bin")
    endif()
  elseif(TAPP_MPI_PATH)
    message(STATUS "TAPP: Could not find mpicc/mpicxx in its 'bin' subdirectory.")
    message(STATUS "TAPP: Forcing MPI search to: ${TAPP_MPI_PATH}")
    find_package(MPI REQUIRED HINTS "${TAPP_MPI_PATH}" NO_DEFAULT_PATH)
  else()
    message(STATUS "TAPP: Using default system path for MPI.")
    find_package(MPI REQUIRED)
  endif()

  # find_package(OpenMP REQUIRED)

  # Paths for CTF and OpenBLAS
  set(CTF_PATH "$ENV{CTF_PATH}" CACHE PATH "Root directory of CTF installation")
  set(OPENBLAS_PATH "/opt/software/gaia/prod/3.2.0/openblas-0.3.26-gcc-12.1.genoa-3utm" CACHE PATH "Root of OpenBLAS install")

  find_path(CTF_INCLUDE_DIR ctf.hpp
    PATHS "${CTF_PATH}/include"
    DOC "CTF include directory")
  
  find_library(CTF_LIBRARY ctf
    PATHS "${CTF_PATH}/lib_shared" 
    DOC "CTF shared library file")

  find_library(OPENBLAS_LIBRARY openblas
    PATHS "${OPENBLAS_PATH}/lib"
    DOC "OpenBLAS library file")

  if(NOT CTF_INCLUDE_DIR OR NOT CTF_LIBRARY OR NOT OPENBLAS_LIBRARY)
    message(FATAL_ERROR "Could not find CTF or OpenBLAS.
    Please set:
      CTF_PATH (points to CTF install, found: ${CTF_PATH})
      OPENBLAS_PATH (points to OpenBLAS install, found: ${OPENBLAS_PATH})")
  endif()

  # Get directory paths for setting rpath
  get_filename_component(CTF_LIB_DIR ${CTF_LIBRARY} DIRECTORY)
  get_filename_component(OPENBLAS_LIB_DIR ${OPENBLAS_LIBRARY} DIRECTORY)

  target_sources(
    tapp
    PRIVATE
      "ctf-bind/ctf-bind.cpp"
    )

    set_source_files_properties(
    "ctf-bind/ctf-bind.cpp"
    PROPERTIES
    INCLUDE_DIRECTORIES
     "${CTF_INCLUDE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/src/tapp"
    )

  target_include_directories(
    tapp
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/ctf-bind"
      "${CMAKE_CURRENT_SOURCE_DIR}/distributed"
  )

  set_property(
          TARGET tapp
          PROPERTY
          CXX_STANDARD 17
          CXX_STANDARD_REQUIRED YES
          CXX_EXTENSIONS NO
  )

  target_compile_definitions(tapp PRIVATE
    ENABLE_CTF=1
    _POSIX_C_SOURCE=200112L
    __STDC_LIMIT_MACROS
    FTN_UNDERSCORE=1
    USE_LAPACK
  )

  target_compile_options(tapp PRIVATE -fopenmp)

  target_link_libraries(
    tapp
    PUBLIC
      ${CTF_LIBRARY}
      ${OPENBLAS_LIBRARY}
      MPI::MPI_CXX
      MPI::MPI_Fortran
      m
  )

  # Add rpath settings from Makefile 
  target_link_options(tapp PRIVATE
    -fopenmp
    "-Wl,-rpath,${CTF_LIB_DIR}"
    "-Wl,-rpath,${OPENBLAS_LIB_DIR}"
  )

  # Add 'worker_ctf' executable
  add_executable(worker_ctf "distributed/worker.cxx")

  target_include_directories(
    worker_ctf
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/distributed"
      ${CTF_INCLUDE_DIR}
  )

  set_property(
    TARGET worker_ctf
    PROPERTY
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
  )

  target_compile_definitions(worker_ctf PRIVATE
    _POSIX_C_SOURCE=200112L
    __STDC_LIMIT_MACROS
    FTN_UNDERSCORE=1
    USE_LAPACK
  )

  target_compile_options(worker_ctf PRIVATE -fopenmp)

  target_link_libraries(
    worker_ctf
    PRIVATE
      ${CTF_LIBRARY}
      ${OPENBLAS_LIBRARY}
      MPI::MPI_CXX
      MPI::MPI_Fortran
      m
  )

  target_link_options(worker_ctf PRIVATE
    -fopenmp
    "-Wl,-rpath,${CTF_LIB_DIR}"
    "-Wl,-rpath,${OPENBLAS_LIB_DIR}"
  )

  set_property(TARGET worker_ctf PROPERTY LINKER_LANGUAGE Fortran)

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

  target_compile_definitions(tapp PRIVATE ENABLE_TBLIS=1)

  target_sources(
  tapp
  PRIVATE
    tblis_bindings/tblis_bind.cpp
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
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tblis_bindings
  )

  set_source_files_properties(
    tblis_bindings/tblis_bind.cpp
    PROPERTIES
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/src/tapp"
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
