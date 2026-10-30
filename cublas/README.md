# TAPP cuBLAS / TTGT bindings

A TAPP back-end that implements `TAPP_execute_product` via the **TTGT**
(transpose-transpose-GEMM-transpose) scheme: tensors are permuted into matrices
with [cuTT](https://github.com/ap-hynninen/cutt), multiplied with cuBLAS
`cublasGemmEx`, and the result permuted back. The TTGT scheme (originally from
the standalone `my-ttgt` project) is integrated directly into the binding, using
TAPP's own structures rather than parallel ones.

The bindings have one TAPP object per file (`attributes`, `datatype`, `error`, `executor`, `runtime`,
`status`, `tensor`) plus the contraction itself in
[`src/product.cu`](src/product.cu), where `TAPP_create_tensor_product` computes
the transpose + GEMM schedule (the TTGT "optimize" step) into the `product_plan`
and `TAPP_execute_product` runs it (the "execute" step). The internal structures
are in [`src/internal.h`](src/internal.h), the public additions to the TAPP
interface in [`include/tapp_cublas.h`](include/tapp_cublas.h).

Each executor owns a CUDA stream and a cuBLAS handle bound to it. Everything an
execution does (cuTT transposes, GEMM, copies, temporary allocations) is queued
on that stream, so `TAPP_execute_product` returns before the result is ready;
use `TAPP_wait_one` / `TAPP_wait_all` or poll the status with
`TAPP_status_get_error`.

## Building

cuBLAS ships with the CUDA toolkit; cuTT does not. Point `CUTT_ROOT` at a cuTT
install prefix (expects `lib/libcutt.a` and `include/cutt.h`), or leave it unset
to have CMake fetch and build cuTT (the
[ValeevGroup fork](https://github.com/ValeevGroup/cutt), which has a CMake build).
The fetched sources are patched with [`cmake/cutt-cuda13.patch`](cmake/cutt-cuda13.patch)
so that they build with CUDA 13, which removed the `clockRate` and
`memoryClockRate` members of `cudaDeviceProp` that cuTT reads; the patch header
explains the change. A cuTT given through `CUTT_ROOT` is used as is, so with
CUDA 13 it must already contain the same fix:

```sh
cmake -B build -DTAPP_REFERENCE_ENABLE_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build
ctest --test-dir build -R tapp-cublas-demo
```

Set `CMAKE_CUDA_ARCHITECTURES` to your target GPU (e.g. `80` for A100, `90` for
H100 / Grace-Hopper). It is required when building on a GPU-less login node; if
left unset it defaults to `80;90` (CUDA 13 dropped Volta, `70`).

`-DTAPP_REFERENCE_ENABLE_CUBLAS=ON` also requires a CUDA compiler and `CUDAToolkit` (CMake finds
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
- **Memory**: `A`, `B`, `C` and `D` must be device pointers. Host memory is not
  copied for you, which is also why the back-end is not part of the multi-tapp
  test (it passes host arrays).
- **Plans are bound to their executor**: the cuTT plans are created on the
  stream of the executor given to `TAPP_create_tensor_product`, so executing
  the plan with another executor returns `TAPP_ERR_PLAN_WRONG_EXECUTOR`.
- **Not implemented**: `TAPP_execute_batched_product` and the
  `TAPP_*_tensor_permute` functions are not provided; multi-tapp reports them
  as unimplemented.

Unsupported contractions return `TAPP_ERR_UNSUPPORTED_DATATYPE` ("Unsupported
datatype for the cuBLAS/TTGT back-end") rather than producing a wrong result.
Failing CUDA, cuBLAS and cuTT calls return their own error code offset by
100000, 200000 and 300000 respectively, which `TAPP_explain_error` describes.

## Attributes

The emulation is configured per executor, through the attributes passed to
`TAPP_create_executor`. The keys and their values are in
[`include/tapp_cublas.h`](include/tapp_cublas.h); every value is an `int`:

| Key | Values | Default | Used by |
|---|---|---|---|
| `TAPP_CUBLAS_EMULATION_STRATEGY` | `TAPP_EMULATION_STRATEGY_*` | `TAPP_DEFAULT_EMULATION_STRATEGY` | Ozaki-I |
| `TAPP_CUBLAS_EMULATION_MANTISSA_CONTROL` | `TAPP_EMULATION_MANTISSA_CONTROL_DYNAMIC` / `_FIXED` | `TAPP_DEFAULT_EMULATION_MANTISSA_CONTROL` | Ozaki-I |
| `TAPP_CUBLAS_EMULATION_NUM_GEMM` | `TAPP_2_GEMM` .. `TAPP_20_GEMM` | `TAPP_DEFAULT_NUM_GEMM` (10) | Ozaki-II |

```c
TAPP_attr attr;
TAPP_create_attr(&attr);
TAPP_num_gemm num_gemm = TAPP_14_GEMM;
TAPP_attr_set(attr, TAPP_CUBLAS_EMULATION_NUM_GEMM, sizeof(num_gemm), &num_gemm);

TAPP_executor exec;
TAPP_create_executor(&exec, attr);
TAPP_destroy_attr(attr);
```

`TAPP_attr_set` rejects unknown keys (`TAPP_ERR_ATTR_UNKNOWN_KEY`) and values
of the wrong size or out of range (`TAPP_ERR_ATTR_INVALID_VALUE`). The
attributes are read when the executor is created; changing them afterwards
does not affect it.

## FP64 emulation (variable precision)

`TAPP_OZAKI_EMULATION_VERSION` selects the emulation path: `0` (default) none,
`1` cuBLAS fixed-point emulation (Ozaki-I), `2` Ozaki-II via GEMMul8.

### Ozaki-I (cuBLAS)

When built with `-DTAPP_OZAKI_EMULATION_VERSION=1` (requires CUDA >= 13), the back-end
can use cuBLAS fixed-point FP64 emulation with a variable mantissa size. The
requested **decimal digits** of precision are selected through the standard
`TAPP_prectype` (the `prec` argument of `TAPP_create_tensor_product`) using the
variable-precision compute types `TAPP_F_<n>_DIGITS` (real) and
`TAPP_C_<n>_DIGITS` (complex), defined in [`include/tapp_cublas.h`](include/tapp_cublas.h). They cover
`n = 3..16` plus a `34`-digit (quad-like) point:

```c
// real, 7 correct decimal digits of accumulation
TAPP_create_tensor_product(&plan, exec, /* ... */, info_D, idx_D, TAPP_F_7_DIGITS);
```

The digit count is converted to a max mantissa bit count
(`ceil(log2(10) * digits)`) and applied to the GEMM. These compute types require
an `F64`/`C64` output (rejected otherwise). The emulation strategy and the
mantissa control come from the executor's [attributes](#attributes). With `TAPP_OZAKI_EMULATION_VERSION`
other than `1` they fall back to plain `FP64`, i.e. full precision, ignoring the
digit reduction.

The values are encoded so the digit count is `value % 1000` (real = `1000 + n`,
complex = `2000 + n`).

### Ozaki-II (GEMMul8)

With `-DTAPP_OZAKI_EMULATION_VERSION=2` the GEMM runs through
[GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8), which needs CUDA >= 12.9.
Build GEMMul8 in place and point `OZAKI_II_ROOT` at it (expects
`lib/libgemmul8.a` and `include/gemmul8.hpp`). `OZAKI_II_COMPUTE_CAPABILITY`
must match the `GPU_ARCH` GEMMul8 was built for:

```sh
make -C /path/to/GEMMul8 BACKEND=cuda GPU_ARCH=90 OPS=gemm
cmake -B build -DTAPP_REFERENCE_ENABLE_CUBLAS=ON -DTAPP_OZAKI_EMULATION_VERSION=2 \
      -DOZAKI_II_ROOT=/path/to/GEMMul8 -DOZAKI_II_COMPUTE_CAPABILITY=90
```

The number of moduli (split GEMMs) is set per executor with the
`TAPP_CUBLAS_EMULATION_NUM_GEMM` [attribute](#attributes). More GEMMs give a
more accurate but slower product; there is no simple relation between the two.

GEMMul8 reports its own failures (e.g. a workspace limit that is too small)
only through `assert`, so in a release build they go unnoticed. Launch errors
of its kernels are still caught and returned as CUDA errors.
