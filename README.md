# TAPP: Tensor Algebra Processing Primitives

[![Linux/MacOS Build](https://github.com/TAPPorg/reference-implementation/actions/workflows/cmake.yml/badge.svg)](https://github.com/TAPPorg/reference-implementation/actions/workflows/cmake.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE.md)

TAPP is a low-level, standard C interface for tensor operations — "a BLAS for tensors". It decouples the application layer from the implementation layer, so that application code can be written once against a stable API and switch between optimized back-ends (e.g. [TBLIS](https://github.com/devinamatthews/tblis), cuTENSOR) easily at link time.

This repository contains 
1. The interface (the API headers) 
2. A reference implementation focusing on correctness rather than performance. 
3. Bindings to high-performance implementations (cuTENSOR, TBLIS).

> 📄 **TAPP is described in this paper: [Tensor Algebra Processing Primitives (TAPP): Towards a Standard for Tensor Operations](https://arxiv.org/abs/2601.07827).** 
> If TAPP is useful for your research or software, please [cite TAPP](#citation).

## The operation

TAPP formulates the tensor contraction as

```
D(idx_D) = alpha * A(idx_A) * B(idx_B) + beta * C(idx_C)
```

where `A`, `B`, `C`, `D` are tensors, `alpha` and `beta` are scalars, and each `idx_*` is an array of index labels in the Einstein-summation convention. The relationship between the label sets determines the operation. For example:

```c
int64_t idx_A[3] = {'a', 'b', 'c'};
int64_t idx_B[4] = {'c', 'd', 'e', 'b'};
int64_t idx_C[3] = {'a', 'd', 'e'};
int64_t idx_D[3] = {'a', 'd', 'e'};   // D[a,d,e] = alpha * A[a,b,c] * B[c,d,e,b] + beta * C[a,d,e]
```

Labels shared between input tensors only (here `b`, `c`) are *contracted*; labels shared between an input and the output are *free*; labels appearing in `A`, `B`, and `D` together form a *Hadamard* (batched) product; and labels appearing in only one input are *reduced*.

## Features

The reference implementation supports:

- **Binary contractions** with free and contracted indices
- **Hadamard / batched products** (indices shared across `A`, `B`, and `D`)
- **Reductions** over indices unique to a single input tensor
- **Diagonals** via repeated labels within a single tensor
- **Element-wise conjugation** (`TAPP_CONJUGATE`) of operands
- **Datatypes**: `float`, `double`, complex float, complex double, and (optionally) 16-bit float / bfloat16
- **Precisions**: `TAPP_DEFAULT_PREC`, `TAPP_F32F32_ACCUM_F32`, `TAPP_F64F64_ACCUM_F64`, `TAPP_F16F16_ACCUM_F16`, `TAPP_F16F16_ACCUM_F32`, `TAPP_BF16BF16_ACCUM_F32`
- **General strided layouts**, including negative, mixed-sign, and zero strides, and zero-mode (scalar) tensors

The full operation semantics, edge cases, and terminology are described in the [TAPP whitepaper](https://arxiv.org/abs/2601.07827).

## Repository layout

| Path | Description |
|------|-------------|
| `api/` | The interface specification: header-only `tapp::api` library (`api/include/tapp/*.h`, umbrella `tapp.h`). This is the standard itself. |
| `reference_implementation/` | `tapp::reference`, the shared-library implementation of the API. Core logic is in `src/product.c`. |
| `test/` | Correctness tests against TBLIS (`test.cpp`) and NumPy `einsum` (`test.py`), plus demo targets and a `find_package` consume test. |

## Building

The project uses CMake. The implementation is C99; a C++20 compiler is only required when TBLIS support is enabled.

```bash
cmake -B build
cd build
make -j
ctest --output-on-failure
```

### CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `TAPP_REFERENCE_ENABLE_TBLIS` | `OFF` | Fetch and build TBLIS (via `FetchContent`) and build the `tapp-reference-test++` suite that validates the reference implementation against TBLIS. Requires a working CXX compiler. |
| `TAPP_REFERENCE_ENABLE_F16` | `OFF` | Enable 16-bit float support. |
| `TAPP_REFERENCE_ENABLE_BF16` | `OFF` | Enable bfloat16 support. |
| `TAPP_BUILD_EXERCISE` | `OFF` | Build the contraction teaching exercise (contains TODOs). |

For example, to build with TBLIS validation, use:

```bash
cmake -B build -DTAPP_REFERENCE_ENABLE_TBLIS=ON
```

## Usage

The API follows a *describe → plan → execute* workflow, so a plan can be created once and re-used across multiple executions. A complete,  commented walk-through is in [`examples/driver/driver.c`](examples/driver/driver.c); the essential steps are:

```c
#include <tapp.h>

/* 1. Describe each tensor's datatype, shape (extents), and memory layout (strides). */
int64_t extents_A[3] = {4, 3, 3}, strides_A[3] = {1, 4, 12};
TAPP_tensor_info info_A;
TAPP_create_tensor_info(&info_A, TAPP_F32, 3, extents_A, strides_A);
/* ... likewise for info_B, info_C, info_D ... */

/* 2. Build a reusable contraction plan from the descriptors and index labels. */
TAPP_handle handle;            /* back-end state (stub in the reference impl) */
TAPP_create_handle(&handle);

int64_t idx_A[3] = {'a', 'b', 'c'};
int64_t idx_B[4] = {'c', 'd', 'e', 'b'};
int64_t idx_C[3] = {'a', 'd', 'e'};
int64_t idx_D[3] = {'a', 'd', 'e'};

TAPP_tensor_product plan;
TAPP_create_tensor_product(&plan, handle,
                           TAPP_IDENTITY, info_A, idx_A,
                           TAPP_IDENTITY, info_B, idx_B,
                           TAPP_IDENTITY, info_C, idx_C,
                           TAPP_IDENTITY, info_D, idx_D,
                           TAPP_DEFAULT_PREC);

/* 3. Execute on actual data pointers. */
TAPP_executor exec;
TAPP_create_executor(&exec);
TAPP_status status;
float alpha = 1.0f, beta = 0.0f;
TAPP_execute_product(plan, exec, &status, &alpha, A, B, &beta, C, D);

TAPP_destroy_tensor_produc(plan);
TAPP_destroy_executor(exec);
TAPP_destroy_handle(handle);
```

API functions return a `TAPP_error` code; use `TAPP_check_success()` to test it and `TAPP_explain_error()` to obtain a human-readable description.

### Using TAPP from another CMake project

Exposes the `tapp::reference` and `tapp::api` targets:

```cmake
find_package(tapp REQUIRED COMPONENTS reference)
target_link_libraries(my_app PRIVATE tapp::reference)
```

See [`test/consume/`](test/consume/) for a working example.

## Examples and exercises

The `examples/` directory contains hands-on exercises for learning the interface:

- **Contraction** — implement a tensor contraction with TAPP (`examples/exercise_contraction/`).
- **Tucker** — integrate TAPP into a practical Tucker-decomposition workflow alongside NumPy/TensorLy (`examples/exercise_tucker/`).

Build and usage instructions, including the required Python dependencies for the Tucker exercise, are in [`examples/README.md`](examples/README.md).

## Background

TAPP was devised following the [CECAM Workshop on Tensor Contraction Library Standardization](https://tensor.sciencesconf.org/) held in Toulouse, May 22–24, 2024. A working group meets regularly to maintain the standard. 

## Contributing

Please submit technical questions, suggestions, and bug reports via [GitHub Issues](https://github.com/TAPPorg/reference-implementation/issues). For other inquiries, contact the maintainers below.

## Authors and contact

Maintained by a working group of application experts, library developers, and hardware vendors. See [AUTHORS.md](AUTHORS.md) for the full list. Primary contacts:

- Devin Matthews — `damatthews at mail.smu.edu`
- Paolo Bientinesi — `pauldj at cs.umu.se`
- Jan Brandejs — `jbrandejs at irsamc.ups-tlse.fr`

## Citation

If TAPP is useful in your research or software, please cite TAPP:

> J. Brandejs, N. Hörnblad, E. F. Valeev, A. Heinecke, J. Hammond, D. Matthews, and P. Bientinesi, *Tensor Algebra Processing Primitives (TAPP): Towards a Standard for Tensor Operations*, [arXiv:2601.07827](https://arxiv.org/abs/2601.07827).

```bibtex
@misc{tapp2026,
  title = {Tensor Algebra Processing Primitives (TAPP): Towards a Standard for Tensor Operations},
  author = {Brandejs,  Jan and H\"{o}rnblad,  Niklas and Valeev,  Edward F. and Heinecke,  Alexander and Hammond,  Jeff and Matthews,  Devin and Bientinesi,  Paolo},
  doi = {10.48550/ARXIV.2601.07827},
  url = {https://arxiv.org/abs/2601.07827},
  publisher = {arXiv},
  year = {2026}
}
```

## License

TAPP is released under the BSD 3-Clause License. See [LICENSE.md](LICENSE.md).
