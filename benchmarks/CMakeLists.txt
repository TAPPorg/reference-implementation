# TAPP benchmarks. Each source is a standalone benchmark using the TAPP API plus
# CUDA runtime for device memory and timing. Add new ones to the list below.

if(TAPP_CUBLAS)
  set(BENCH_BACKEND tapp::cublas)
elseif(TAPP_CUTENSOR)
  set(BENCH_BACKEND tapp::cutensor)
else()
  message(FATAL_ERROR "TAPP_BENCHMARKS needs a GPU back-end (TAPP_CUBLAS or TAPP_CUTENSOR)")
endif()

set(TAPP_BENCHMARK_SOURCES
  ccsd_bottlenecks
)

foreach(bench ${TAPP_BENCHMARK_SOURCES})
  add_executable(bench-${bench} ${bench}.cpp)
  set_target_properties(bench-${bench} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
  target_link_libraries(bench-${bench} PRIVATE ${BENCH_BACKEND} CUDA::cudart)
endforeach()
