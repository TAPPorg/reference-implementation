cmake_minimum_required(VERSION 3.15)
project(tapp-consume-test LANGUAGES C)

find_package(tapp REQUIRED COMPONENTS reference)

add_executable(smoke smoke.c)
target_link_libraries(smoke PRIVATE tapp::reference)

enable_testing()
add_test(NAME smoke COMMAND smoke)
