#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_TENSOR_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_TENSOR_H_

#include <tapp/tensor.h>

#include <cstdint>

#include "error.h"
#include "handle.h"
#include "datatype.h"

// Logical/physical description of a dense tensor. The TTGT back-end works on
// dense (generalized column-major) layouts; `extents`/`strides` are retained
// both for the host<->device copy bounds and to feed the TTGT plan.
struct tensor_info
{
    int nmode;
    int64_t* extents;
    int64_t* strides;
    size_t elements;
    size_t copy_size;     // bytes spanned by the data in memory
    int64_t data_offset;  // byte offset of element 0 (for negative strides)
    TAPP_datatype type;
};

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_TENSOR_H_ */
