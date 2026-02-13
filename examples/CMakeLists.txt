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
    tapp::reference
  )

add_test(
  NAME tapp-reference-driver
  COMMAND $<TARGET_FILE:tapp-reference-driver>
  )

# ----------------------------------------------------------------------------
# exercise: contraction

if(TAPP_BUILD_EXERCISE)
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
      tapp::reference
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
    tapp::reference
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
    tapp::reference
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
