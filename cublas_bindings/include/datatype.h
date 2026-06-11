#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_DATATYPE_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_DATATYPE_H_

#include <tapp/datatype.h>

#include <cublas_v2.h>
#include <library_types.h>

#include <complex>

// Translate a TAPP computational precision into the cuBLAS compute type passed
// to cublasGemmEx. `datatype` resolves TAPP_DEFAULT_PREC.
cublasComputeType_t translate_prectype(TAPP_prectype prec, TAPP_datatype datatype);

// CUDA data type (CUDA_R_32F, ...) for a TAPP storage datatype, for cublasGemmEx.
cudaDataType get_cuda_datatype(TAPP_datatype type);

size_t sizeof_datatype(TAPP_datatype type);

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_DATATYPE_H_ */
