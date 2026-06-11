#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_HANDLE_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_HANDLE_H_

#include <tapp/handle.h>

#include <cublas_v2.h>

#include "error.h"

// The TAPP library handle owns the back-end state: a cuBLAS handle (created once
// here rather than per tensor-product plan) and the implementation attribute
// store.
struct handle
{
    cublasHandle_t cublas;
    intptr_t* attributes;
};

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_HANDLE_H_ */
