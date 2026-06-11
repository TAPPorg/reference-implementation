#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_DATATYPE_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_DATATYPE_H_

#include <tapp/datatype.h>

#include <cublas_v2.h>

#include <complex>

#include "../ttgt/ttgt_utils.h"

// Translate a TAPP storage datatype into the TTGT DataType enum used by the
// vendored TTGT implementation. Only F32/F64/C32/C64 are supported; F16/BF16
// have no TTGT/cuBLAS-GEMM equivalent here.
DataType translate_datatype(TAPP_datatype type);

// Translate a TAPP computational precision into the cuBLAS compute type passed
// to cublasGemmEx. `datatype` is used to resolve TAPP_DEFAULT_PREC.
cublasComputeType_t translate_prectype(TAPP_prectype prec, TAPP_datatype datatype);

size_t sizeof_datatype(TAPP_datatype type);

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_DATATYPE_H_ */
