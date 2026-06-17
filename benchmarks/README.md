# TAPP benchmarks

GPU benchmarks using the TAPP API. Each `*.cpp` is standalone; timing covers
only `TAPP_execute_product` (allocation/fill excluded). Storage and compute
datatypes are set near the top of each file.

```sh
cmake -B build -DTAPP_BENCHMARKS=ON -DTAPP_CUBLAS=ON \
      -DCUTT_ROOT=/path/to/cutt -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --target bench-ccsd_bottlenecks
./build/benchmarks/bench-ccsd_bottlenecks [nocc nvirt [nocc_PH nvirt_PH]]
```

Back-end: `tapp::cublas` if `TAPP_CUBLAS`, else `tapp::cutensor`. Add a benchmark
by dropping a source file here and listing it in `CMakeLists.txt`.
