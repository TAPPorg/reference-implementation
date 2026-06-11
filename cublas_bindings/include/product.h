#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_PRODUCT_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_PRODUCT_H_

#include <tapp/product.h>

#include <cublas_v2.h>
#include <cutt.h>

#include <cstdint>

#include "error.h"
#include "handle.h"
#include "tensor.h"
#include "attributes.h"
#include "datatype.h"

// A built tensor-contraction plan for the cuBLAS/TTGT back-end.
//
// The TTGT (transpose-transpose-GEMM-transpose) schedule is computed once in
// TAPP_create_tensor_product and replayed in TAPP_execute_product: each input
// is optionally permuted into a matrix with cuTT, the matrices are multiplied
// with cuBLAS GEMM, and the result is optionally permuted back. Everything the
// execution needs is stored here directly (no separate plan/info objects).
struct product_plan
{
    TAPP_handle handle;  // library handle (cuBLAS handle + attributes)
    bool failed;         // set when the contraction is outside TTGT's support

    // Transpose plans. transposeX == false means tensor X is already in the
    // layout the GEMM expects, so it is fed to the GEMM unchanged.
    bool transposeA, transposeB, transposeC;
    cuttHandle planA, planB, planC;

    // GEMM configuration (CT = alpha * op(A) * op(B)).
    cublasOperation_t transa, transb;
    int m, n, k, lda, ldb, ldc;
    cublasComputeType_t compute_type;

    // Requested decimal digits of precision for cuBLAS fixed-point FP64
    // emulation (0 = emulation off). Only meaningful when built with EMULATION
    // and for F64/C64 outputs; converted to a max mantissa bit count at execute.
    int prec_digits;

    // Element counts, for sizing the transposed device buffers.
    size_t elements_A, elements_B, elements_D;
    TAPP_datatype type_A, type_B, type_D;

    // Host<->device transfer description per operand.
    size_t copy_size_A;  int64_t data_offset_A;
    size_t copy_size_B;  int64_t data_offset_B;
    size_t copy_size_C;  int64_t data_offset_C;
    size_t copy_size_D;  int64_t data_offset_D;

    TAPP_element_op op_D;
};

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_PRODUCT_H_ */
