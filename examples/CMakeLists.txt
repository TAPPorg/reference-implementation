# Examples and exercises, built as part of the testing of the project

# Next to the tests and libraries in the build directory instead of in build/examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# ----------------------------------------------------------------------------
# driver

add_executable(tapp-reference-driver)

target_sources(
  tapp-reference-driver
  PRIVATE
    driver/driver.c
    ${PROJECT_SOURCE_DIR}/test/helpers.c
    ${PROJECT_SOURCE_DIR}/test/helpers.h
  )

target_include_directories(
  tapp-reference-driver
  PRIVATE
    ${PROJECT_SOURCE_DIR}/test
  )

target_link_libraries(
  tapp-reference-driver
  PRIVATE
    tapp-reference
  )

add_test(
  NAME tapp-reference-driver
  COMMAND $<TARGET_FILE:tapp-reference-driver>
  )

# ----------------------------------------------------------------------------
# cuTENSOR: linked directly to the cuTENSOR implementation

if(TAPP_REFERENCE_ENABLE_CUTENSOR)
  add_executable(tapp-cutensor-example)

  target_sources(
    tapp-cutensor-example
    PRIVATE
      cutensor/cutensor.c
    )

  target_include_directories(
    tapp-cutensor-example
    PRIVATE
      ${PROJECT_SOURCE_DIR}/cutensor/include
    )

  target_link_libraries(
    tapp-cutensor-example
    PRIVATE
      tapp::cutensor
      CUDA::cudart
    )

  add_test(
    NAME tapp-cutensor-example
    COMMAND $<TARGET_FILE:tapp-cutensor-example>
    )

  # Two executors, each on a CUDA stream of its own
  add_executable(tapp-cutensor-two-exec)

  target_sources(
    tapp-cutensor-two-exec
    PRIVATE
      cutensor/cutensor_two_exec.c
    )

  target_include_directories(
    tapp-cutensor-two-exec
    PRIVATE
      ${PROJECT_SOURCE_DIR}/cutensor/include
    )

  target_link_libraries(
    tapp-cutensor-two-exec
    PRIVATE
      tapp::cutensor
      CUDA::cudart
    )
endif()

# ----------------------------------------------------------------------------
# TBLIS: linked directly to the TBLIS implementation

# The CPU masks of the examples are only set on Linux, elsewhere the executors run on any CPU
if(TAPP_REFERENCE_ENABLE_TBLIS)
  add_executable(tapp-tblis-example)

  target_sources(
    tapp-tblis-example
    PRIVATE
      tblis/tblis.c
    )

  target_link_libraries(
    tapp-tblis-example
    PRIVATE
      tapp-tblis
    )

  add_test(
    NAME tapp-tblis-example
    COMMAND $<TARGET_FILE:tapp-tblis-example>
    )

  # Two executors, each on CPUs of its own
  add_executable(tapp-tblis-two-exec)

  target_sources(
    tapp-tblis-two-exec
    PRIVATE
      tblis/tblis_two_exec.c
    )

  target_link_libraries(
    tapp-tblis-two-exec
    PRIVATE
      tapp-tblis
    )
endif()

# ----------------------------------------------------------------------------
# multi-tapp: the implementations are loaded at runtime, so only their headers are used

# Two TBLIS executors, each on CPUs of its own. The CPU masks are only set on Linux
if(TAPP_REFERENCE_ENABLE_TBLIS)
  add_executable(multi-tapp-tblis-two-exec)

  target_sources(
    multi-tapp-tblis-two-exec
    PRIVATE
      multi-tapp/multi-tapp_tblis_two_exec.c
    )

  target_include_directories(
    multi-tapp-tblis-two-exec
    PRIVATE
      $<TARGET_PROPERTY:tapp-tblis,INTERFACE_INCLUDE_DIRECTORIES>
    )

  target_compile_definitions(
    multi-tapp-tblis-two-exec
    PRIVATE
      TAPP_TBLIS_LIBPATH="$<TARGET_FILE:tapp-tblis>"
      TAPP_REFERENCE_LIBPATH="$<TARGET_FILE:tapp-reference>"
    )

  target_link_libraries(
    multi-tapp-tblis-two-exec
    PRIVATE
      multi-tapp
    )

  add_dependencies(multi-tapp-tblis-two-exec tapp-tblis tapp-reference)
endif()

# Two cuTENSOR executors, each on a CUDA stream of its own
if(TAPP_REFERENCE_ENABLE_CUTENSOR)
  add_executable(multi-tapp-cutensor-two-exec)

  target_sources(
    multi-tapp-cutensor-two-exec
    PRIVATE
      multi-tapp/multi-tapp_cutensor_two_exec.c
    )

  target_include_directories(
    multi-tapp-cutensor-two-exec
    PRIVATE
      ${PROJECT_SOURCE_DIR}/cutensor/include
    )

  target_compile_definitions(
    multi-tapp-cutensor-two-exec
    PRIVATE
      TAPP_CUTENSOR_LIBPATH="$<TARGET_FILE:tapp-cutensor>"
      TAPP_REFERENCE_LIBPATH="$<TARGET_FILE:tapp-reference>"
    )

  target_link_libraries(
    multi-tapp-cutensor-two-exec
    PRIVATE
      multi-tapp
      CUDA::cudart
    )

  add_dependencies(multi-tapp-cutensor-two-exec tapp-cutensor tapp-reference)
endif()

# TBLIS and cuTENSOR side by side: one product on the CPU while another runs on the GPU
if(TAPP_REFERENCE_ENABLE_TBLIS AND TAPP_REFERENCE_ENABLE_CUTENSOR)
  add_executable(multi-tapp-cutensor-tblis)

  target_sources(
    multi-tapp-cutensor-tblis
    PRIVATE
      multi-tapp/multi-tapp_cutensor_tblis.c
    )

  # For the attribute keys in tapp_tblis.h and tapp_cutensor.h: the implementations are loaded at run time, not linked
  target_include_directories(
    multi-tapp-cutensor-tblis
    PRIVATE
      ${PROJECT_SOURCE_DIR}/tblis/include
      ${PROJECT_SOURCE_DIR}/cutensor/include
    )

  target_compile_definitions(
    multi-tapp-cutensor-tblis
    PRIVATE
      TAPP_TBLIS_LIBPATH="$<TARGET_FILE:tapp-tblis>"
      TAPP_CUTENSOR_LIBPATH="$<TARGET_FILE:tapp-cutensor>"
      TAPP_REFERENCE_LIBPATH="$<TARGET_FILE:tapp-reference>"
    )

  target_link_libraries(
    multi-tapp-cutensor-tblis
    PRIVATE
      multi-tapp
      CUDA::cudart
    )

  add_dependencies(multi-tapp-cutensor-tblis tapp-tblis tapp-cutensor tapp-reference)
endif()

# ----------------------------------------------------------------------------
# CCD

# Linux only: the example places its executors with CPU masks (cpu_set_t)
if(TAPP_REFERENCE_ENABLE_TBLIS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(tapp-reference-ccd)

  target_sources(
    tapp-reference-ccd
    PRIVATE
      CCD/run_file.c
      CCD/run_file.h
      CCD/ccd.c
    )

  # Only the header of tapp-tblis is used: the library is loaded at runtime
  target_include_directories(
    tapp-reference-ccd
    PRIVATE
      $<TARGET_PROPERTY:tapp-tblis,INTERFACE_INCLUDE_DIRECTORIES>
    )

  target_compile_definitions(
    tapp-reference-ccd
    PRIVATE
      TAPP_TBLIS_LIBPATH="$<TARGET_FILE:tapp-tblis>"
    )

  target_link_libraries(
    tapp-reference-ccd
    PRIVATE
      multi-tapp
    )

  add_dependencies(tapp-reference-ccd tapp-tblis)

  # Script to generate input tensors
  find_package(OpenMP REQUIRED)
  add_executable(tapp-tblis-ccd-reference)

  target_sources(
    tapp-tblis-ccd-reference
    PRIVATE
      CCD/tblis/reference/main.cpp
    )

  target_include_directories(
    tapp-tblis-ccd-reference
    PRIVATE
      $<TARGET_PROPERTY:tapp-tblis,INTERFACE_INCLUDE_DIRECTORIES>
    )
  target_link_libraries(
    tapp-tblis-ccd-reference
    PRIVATE
      tapp-tblis
      OpenMP::OpenMP_CXX
    )

endif()

if(TAPP_REFERENCE_ENABLE_CUTENSOR)
  # Script to compute solution with 1 GPU and 1 stream tensors
  find_package(OpenMP REQUIRED)
  add_executable(tapp-cutensor-ccd)

  target_sources(
    tapp-cutensor-ccd
    PRIVATE
      CCD/cutensor/1-gpu-1-stream/main.cpp
    )

  target_include_directories(
    tapp-cutensor-ccd
    PRIVATE
      $<TARGET_PROPERTY:tapp-cutensor,INTERFACE_INCLUDE_DIRECTORIES>
    )
  target_link_libraries(
    tapp-cutensor-ccd
    PRIVATE
      tapp-cutensor
      CUDA::cudart
      OpenMP::OpenMP_CXX
    )
  # Script to compute solution with 1 GPU and 2 streams tensors
  add_executable(tapp-cutensor-ccd-2-streams)

  target_sources(
    tapp-cutensor-ccd-2-streams
    PRIVATE
      CCD/cutensor/1-gpu-2-stream/main.cpp
    )

  target_include_directories(
    tapp-cutensor-ccd-2-streams
    PRIVATE
      $<TARGET_PROPERTY:tapp-cutensor,INTERFACE_INCLUDE_DIRECTORIES>
    )
  target_link_libraries(
    tapp-cutensor-ccd-2-streams
    PRIVATE
      tapp-cutensor
      CUDA::cudart
      OpenMP::OpenMP_CXX
    )
  # Script to compute solution with 2 GPUs and 2 streams tensors
  add_executable(tapp-cutensor-ccd-2-gpus-2-streams)

  target_sources(
    tapp-cutensor-ccd-2-gpus-2-streams
    PRIVATE
      CCD/cutensor/2-gpu-2-stream/main.cpp
    )

  target_include_directories(
    tapp-cutensor-ccd-2-gpus-2-streams
    PRIVATE
      $<TARGET_PROPERTY:tapp-cutensor,INTERFACE_INCLUDE_DIRECTORIES>
    )
  target_link_libraries(
    tapp-cutensor-ccd-2-gpus-2-streams
    PRIVATE
      tapp-cutensor
      CUDA::cudart
      OpenMP::OpenMP_CXX
    )
endif()

# ----------------------------------------------------------------------------
# exercise: contraction

if(TAPP_REFERENCE_BUILD_EXERCISE)
  add_executable(tapp-reference-exercise_contraction)

  target_sources(
    tapp-reference-exercise_contraction
    PRIVATE
      exercise_contraction/exercise_contraction.c
      ${PROJECT_SOURCE_DIR}/test/helpers.c
      ${PROJECT_SOURCE_DIR}/test/helpers.h
    )

  target_include_directories(
    tapp-reference-exercise_contraction
    PRIVATE
      ${PROJECT_SOURCE_DIR}/test
    )

  target_link_libraries(
    tapp-reference-exercise_contraction
    PRIVATE
      tapp-reference
    )

  add_test(
    NAME tapp-reference-exercise_contraction
    COMMAND $<TARGET_FILE:tapp-reference-exercise_contraction>
    )
endif()

# ----------------------------------------------------------------------------
# exercise: contraction answers

add_executable(tapp-reference-exercise_contraction_answers)

target_sources(
  tapp-reference-exercise_contraction_answers
  PRIVATE
    exercise_contraction/answers/exercise_contraction_answers.c
    ${PROJECT_SOURCE_DIR}/test/helpers.c
    ${PROJECT_SOURCE_DIR}/test/helpers.h
  )

target_include_directories(
  tapp-reference-exercise_contraction_answers
  PRIVATE
    ${PROJECT_SOURCE_DIR}/test
  )

target_link_libraries(
  tapp-reference-exercise_contraction_answers
  PRIVATE
    tapp-reference
  )

add_test(
  NAME tapp-reference-exercise_contraction_answers
  COMMAND $<TARGET_FILE:tapp-reference-exercise_contraction_answers>
  )

# ----------------------------------------------------------------------------
# exercise: tucker

add_library(tapp-reference-exercise_tucker SHARED)

target_sources(
  tapp-reference-exercise_tucker
  PUBLIC
    exercise_tucker/tapp_tucker/exercise_tucker.h
  PRIVATE
    exercise_tucker/tapp_tucker/exercise_tucker.c
  )

target_link_libraries(
  tapp-reference-exercise_tucker
  PRIVATE
    tapp-reference
  )

# ----------------------------------------------------------------------------
# exercise: tucker answers

add_library(tapp-reference-exercise_tucker_answers SHARED)

target_sources(
  tapp-reference-exercise_tucker_answers
  PUBLIC
    exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.h
  PRIVATE
    exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.c
  )

target_link_libraries(
  tapp-reference-exercise_tucker_answers
  PRIVATE
    tapp-reference
  )
