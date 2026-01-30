#ifndef TAPP_REF_IMPL_MULTI_TAPP_PRODUCT_H_
#define TAPP_REF_IMPL_MULTI_TAPP_PRODUCT_H_

#include "tapp/product.h"
#include "executor.h"
#include "handle.h"
#include "status.h"
#include "tensor.h"

struct Multi_TAPP_tensor_product
{
    uint64_t impl_id;
    TAPP_tensor_product* plan;
};

#endif /* TAPP_REF_IMPL_MULTI_TAPP_PRODUCT_H_ */