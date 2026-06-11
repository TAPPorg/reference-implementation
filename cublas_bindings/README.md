# TAPP cuBLAS / TTGT bindings

A TAPP back-end that implements `TAPP_execute_product` via the **TTGT**
(transpose-transpose-GEMM-transpose) scheme: tensors are permuted into matrices
with [cuTT](https://github.com/ap-hynninen/cutt), multiplied with cuBLAS
`cublasGemmEx`, and the result permuted back. The TTGT scheme (originally from
the standalone `my-ttgt` project) is integrated directly into the binding, using
TAPP's own structures rather than parallel ones.

The layout mirrors [`../cutensor_bindings`](../cutensor_bindings): one
TAPP object per file (`handle`, `tensor`, `executor`, `attributes`, `status`,
`error`, `datatype`) plus the contraction itself in
[`src/product.cu`](src/product.cu), where `TAPP_create_tensor_product` computes
the transpose + GEMM schedule (the TTGT "optimize" step) into the `product_plan`
and `TAPP_execute_product` runs it (the "execute" step). The cuBLAS handle lives
in the TAPP library handle.

## Building

cuBLAS ships with the CUDA toolkit; cuTT does not. Point `CUTT_ROOT` at a cuTT
install prefix (expects `lib/libcutt.a` and `include/cutt.h`):

```sh
cmake -B build -DTAPP_CUBLAS=ON -DCUTT_ROOT=/path/to/cutt -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build
ctest --test-dir build -R tapp-cublas-demo
```

Set `CMAKE_CUDA_ARCHITECTURES` to your target GPU (e.g. `80` for A100, `90` for
H100 / Grace-Hopper). It is required when building on a GPU-less login node; if
left unset it defaults to a portable `70;80;90` list.

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
- **Execution is synchronous**: the GEMM and cuTT transposes run on the default
  stream, so transfers block (`cudaDeviceSynchronize`) and the executor stream is
  used only to record the status object.

Unsupported inputs return TAPP error code 16 ("Unsupported datatype for the
cuBLAS/TTGT back-end") rather than producing a wrong result.

## FP64 emulation (variable precision)

When built with `-DTAPP_CUBLAS_EMULATION=1` (requires CUDA >= 13), the back-end
can use cuBLAS fixed-point FP64 emulation with a variable mantissa size. The
caller requests a number of **decimal digits** of precision via the
`ATTR_KEY_PRECISION_DIGITS` (= 1) attribute on the handle before creating a plan:

```c
int digits = 7;
TAPP_attr_set(handle, ATTR_KEY_PRECISION_DIGITS, &digits);
TAPP_create_tensor_product(&plan, handle, /* ... */);  // captures the setting
```

The digit count is converted to a max mantissa bit count
(`ceil(log2(10) * digits)`) and applied to the GEMM. It only affects `F64`/`C64`
outputs; for other datatypes (or when `digits == 0`, the default) the normal
compute type is used. Without `EMULATION` the attribute is accepted but ignored.
