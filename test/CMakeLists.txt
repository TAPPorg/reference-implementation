# ----------------------------------------------------------------------------
# TBLIS test

if(TAPP_REFERENCE_USE_TBLIS)
  add_executable(tapp-reference-test)

  target_sources(
    tapp-reference-test
    PRIVATE
      test.cpp
      test.h
    )

  target_link_libraries(
    tapp-reference-test
    PRIVATE
      tapp::reference
      tblis-static
    )

  set_property(
    TARGET tapp-reference-test
    PROPERTY
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
  )

  add_test(
    NAME tapp-reference-test
    COMMAND $<TARGET_FILE:tapp-reference-test>
    )
endif()

# ----------------------------------------------------------------------------
# demo

add_executable(tapp-reference-demo)

target_sources(
  tapp-reference-demo
  PRIVATE
    demo.c
    helpers.c
    helpers.h
  )

target_link_libraries(
  tapp-reference-demo
  PRIVATE
    tapp::reference
  )

add_test(
  NAME tapp-reference-demo
  COMMAND $<TARGET_FILE:tapp-reference-demo>
  )

# ----------------------------------------------------------------------------
# cutensor specific code

if (TAPP_CUTENSOR)
  # ----------------------------------------------------------------------------
  # cutensor demo

  add_executable(tapp-cutensor-demo)

  target_sources(
    tapp-cutensor-demo
    PRIVATE
      cutensor_demo.cpp
      helpers.c
      helpers.h
  )

  target_link_libraries(
    tapp-cutensor-demo
    PRIVATE
      tapp::cutensor
      CUDA::cudart
  )

  target_include_directories(
    tapp-cutensor-demo
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(
    NAME tapp-cutensor-demo
    COMMAND $<TARGET_FILE:tapp-cutensor-demo>
  )

endif()

# ----------------------------------------------------------------------------
# cublas specific code

if (TAPP_CUBLAS)
  # ----------------------------------------------------------------------------
  # cublas demo

  add_executable(tapp-cublas-demo)

  target_sources(
    tapp-cublas-demo
    PRIVATE
      cublas_demo.cpp
  )

  target_link_libraries(
    tapp-cublas-demo
    PRIVATE
      tapp::cublas
      CUDA::cudart
  )

  target_include_directories(
    tapp-cublas-demo
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(
    NAME tapp-cublas-demo
    COMMAND $<TARGET_FILE:tapp-cublas-demo>
  )
endif()

# ----------------------------------------------------------------------------
# dynamic-launch demo and test
#
# These load TAPP implementations from shared libraries at runtime (dlopen)
# rather than linking them. The test is the cross-implementation check: it loads
# two backends and compares their results. They are therefore built whenever any
# non-reference backend is available, and a comparison is registered per enabled
# backend (reference vs. that backend). The library to use is passed on the
# command line, so the binaries are backend-agnostic.

if (TAPP_CUTENSOR OR TAPP_CUBLAS)
  # --------------------------------------------------------------------------
  # demo using a dynamically loaded library (single implementation)

  add_executable(tapp-reference-demo-dynamic)

  target_compile_definitions(
    tapp-reference-demo-dynamic
    PRIVATE
      TAPP_DYNAMIC_LAUNCH
  )

  target_sources(
    tapp-reference-demo-dynamic
    PRIVATE
      demo.c
      helpers.c
      helpers.h
  )

  target_link_libraries(
    tapp-reference-demo-dynamic
    PRIVATE
      tapp::api
      ${CMAKE_DL_LIBS}
  )

  # --------------------------------------------------------------------------
  # test comparing two dynamically loaded implementations

  add_executable(tapp-reference-test-dynamic)

  target_compile_definitions(
    tapp-reference-test-dynamic
    PRIVATE
      TAPP_DYNAMIC_LAUNCH
  )

  target_sources(
    tapp-reference-test-dynamic
    PRIVATE
      test.cpp
      test.h
  )

  target_link_libraries(
    tapp-reference-test-dynamic
    PRIVATE
      tapp::api
      ${CMAKE_DL_LIBS}
  )

  # Per-backend registrations: run the demo against each backend, and compare
  # the reference implementation against each backend.
  if (TAPP_CUTENSOR)
    add_test(
      NAME tapp-cutensor-demo-dynamic
      COMMAND $<TARGET_FILE:tapp-reference-demo-dynamic> $<TARGET_FILE:tapp-cutensor>
    )
    add_test(
      NAME tapp-cutensor-test-dynamic
      COMMAND $<TARGET_FILE:tapp-reference-test-dynamic> $<TARGET_FILE:tapp-reference> $<TARGET_FILE:tapp-cutensor>
    )
  endif()

  if (TAPP_CUBLAS)
    add_test(
      NAME tapp-cublas-demo-dynamic
      COMMAND $<TARGET_FILE:tapp-reference-demo-dynamic> $<TARGET_FILE:tapp-cublas>
    )
    add_test(
      NAME tapp-cublas-test-dynamic
      COMMAND $<TARGET_FILE:tapp-reference-test-dynamic> $<TARGET_FILE:tapp-reference> $<TARGET_FILE:tapp-cublas>
    )
  endif()
endif()
