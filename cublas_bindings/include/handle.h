#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_HANDLE_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_HANDLE_H_

#include <tapp/handle.h>

#include "error.h"

// The cuBLAS/TTGT back-end does not need a persistent library handle (the
// vendored TTGTPlan creates its own cublasHandle_t per plan). The handle only
// carries the implementation attribute store, mirroring the cuTENSOR bindings.
struct handle
{
    intptr_t* attributes;
};

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_HANDLE_H_ */
