#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_TENSOR_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_TENSOR_H_

#include <tapp/tensor.h>

#include <cutensor.h>

#include <cstring>

#include "error.h"
#include "handle.h"
#include "datatype.h"

struct tensor_info
{
    int nmode;
    int64_t *extents;
    int64_t *strides;
    size_t elements;
    size_t copy_size;
    int64_t data_offset;
    TAPP_datatype type;
    cutensorTensorDescriptor_t* desc;
};

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_TENSOR_H_ */