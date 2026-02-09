#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_

#include <tapp/product.h>

#include <cutensor.h>

#include <vector>
#include <algorithm>
#include <assert.h>

#include "error.h"
#include "handle.h"
#include "tensor.h"

struct product_plan
{
    int64_t data_offset_A;
    size_t copy_size_A;
    int64_t data_offset_B;
    size_t copy_size_B;
    int64_t data_offset_C;
    size_t copy_size_C;
    int64_t data_offset_D;
    size_t copy_size_D;
    int64_t sections_D;
    int64_t section_size_D;
    int64_t sections_nmode_D;
    int64_t* section_extents_D;
    int64_t* section_strides_D;
    TAPP_datatype type_D;
    cutensorPlan_t* contraction_plan;
    cutensorPlan_t* permutation_plan;
    cutensorHandle_t* handle;
};

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_ */