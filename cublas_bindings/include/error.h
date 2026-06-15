#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_ERROR_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_ERROR_H_

#include <tapp/error.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>

inline TAPP_error tapp_error(cudaError_t e)    { return e == cudaSuccess ? TAPP_SUCCESS : tapp_error(TAPP_ERROR_TYPE_CUDA, (int)e); }
inline TAPP_error tapp_error(cublasStatus_t e) { return e == CUBLAS_STATUS_SUCCESS ? TAPP_SUCCESS : tapp_error(TAPP_ERROR_TYPE_CUBLAS, (int)e); }

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_ERROR_H_ */
