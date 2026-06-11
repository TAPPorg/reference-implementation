# TAPP cuBLAS / TTGT bindings

A TAPP back-end that implements `TAPP_execute_product` via the **TTGT**
(transpose-transpose-GEMM-transpose) scheme: tensors are permuted into matrices
with [cuTT](https://github.com/ap-hynninen/cutt), multiplied with cuBLAS
`cublasGemmEx`, and the result permuted back. The TTGT implementation under
[`ttgt/`](ttgt/) is vendored from the standalone `my-ttgt` project.

The layout mirrors [`../cutensor_bindings`](../cutensor_bindings): one
TAPP object per file (`handle`, `tensor`, `executor`, `attributes`, `status`,
`error`, `datatype`) plus the contraction itself in
[`src/product.cu`](src/product.cu), where `TAPP_create_tensor_product` builds a
`TTGTPlan` (`optimize`) and `TAPP_execute_product` runs it (`execute`).

## Building

cuBLAS ships with the CUDA toolkit; cuTT does not. Point `CUTT_ROOT` at a cuTT
install prefix (expects `lib/libcutt.a` and `include/cutt.h`):

```sh
cmake -B build -DTAPP_CUBLAS=ON -DCUTT_ROOT=/path/to/cutt
cmake --build build
ctest --test-dir build -R tapp-cublas-demo
```

`-DTAPP_CUBLAS=ON` also requires a CUDA compiler and `CUDAToolkit` (CMake finds
`CUDA::cudart` / `CUDA::cublas`). The demo lives in
[`../test/cublas_demo.cpp`](../test/cublas_demo.cpp).

## Scope and limitations

This is a first TTGT-based back-end and is intentionally narrower than the
cuTENSOR bindings:

- **Operations**: plain (Case 1/2) contractions only. Repeated/isolated indices
  (Cases 3–5) are not supported.
- **Element-wise ops**: identity only; conjugation (`TAPP_CONJUGATE`) is
  rejected.
- **Layout**: dense, generalized **column-major** tensors. TTGT works on
  contiguous data and does not honour arbitrary / negative strides the way the
  cuTENSOR binding does.
- **Datatypes**: `F32`, `F64`, `C32`, `C64`. `F16` / `BF16` are rejected.
- **Execution is synchronous**: TTGT runs on the default stream with its own
  cuBLAS handle, so transfers block and the executor stream is used only to
  record the status object.

Unsupported inputs return TAPP error code 16 ("Unsupported datatype for the
cuBLAS/TTGT back-end") rather than producing a wrong result.
