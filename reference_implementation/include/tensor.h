#ifndef TAPP_REF_IMPL_REF_IMPL_TENSOR_H_
#define TAPP_REF_IMPL_REF_IMPL_TENSOR_H_

#include <tapp/tensor.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tensor_info
{
    TAPP_datatype type;
    int nmode;
    int64_t* extents;
    int64_t* strides;
};

#endif  /* TAPP_REF_IMPL_REF_IMPL_TENSOR_H_ */