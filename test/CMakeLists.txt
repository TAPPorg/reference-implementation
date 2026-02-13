# ----------------------------------------------------------------------------
# TBLIS test

if(TAPP_REFERENCE_USE_TBLIS)
  add_executable(tapp-reference-test++)

  target_sources(
    tapp-reference-test++
    PRIVATE
      test.cpp
      test.h
    )

  target_link_libraries(
    tapp-reference-test++
    PRIVATE
      tapp::reference
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

  # ----------------------------------------------------------------------------
  # demo using dynamic library

  add_executable(tapp-reference-demo-dynamic)

  target_sources(
    tapp-reference-demo-dynamic
    PRIVATE
      demo_dynamic.c
      helpers.c
      helpers.h
  )

  target_link_libraries(
    tapp-reference-demo-dynamic
    PRIVATE
      tapp::api
      ${CMAKE_DL_LIBS}
  )

  add_test(
    NAME tapp-reference-demo-dynamic
    COMMAND $<TARGET_FILE:tapp-reference-demo-dynamic>
  )

  # ----------------------------------------------------------------------------
  # test using dynamic library

  add_executable(tapp-reference-test-dynamic)

  target_sources(
    tapp-reference-test-dynamic
    PRIVATE
      test_dynamic.cpp
      test_dynamic.h
  )

  target_link_libraries(
    tapp-reference-test-dynamic
    PRIVATE
      tapp::api
      ${CMAKE_DL_LIBS}
  )

  add_test(
    NAME tapp-reference-test-dynamic
    COMMAND $<TARGET_FILE:tapp-reference-test-dynamic>
  )

endif()
