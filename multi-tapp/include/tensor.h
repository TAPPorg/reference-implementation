#ifndef TAPP_REF_IMPL_MULTI_TAPP_TENSOR_H_
#define TAPP_REF_IMPL_MULTI_TAPP_TENSOR_H_

#include "tapp/tensor.h"
#include "handle.h"

struct Multi_TAPP_tensor_info
{
    uint64_t impl_id;
    TAPP_tensor_info* info;
};

#endif /* TAPP_REF_IMPL_MULTI_TAPP_TENSOR_H_ */