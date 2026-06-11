#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_ERROR_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_ERROR_H_

#include <tapp/error.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>

int pack_error(int current_value, int tapp_err);
int pack_error(int current_value, cublasStatus_t e);
int pack_error(int current_value, cudaError_t e);

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_ERROR_H_ */
