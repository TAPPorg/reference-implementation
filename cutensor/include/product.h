#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_

#include <tapp/product.h>

#include <cutensor.h>

#include <vector>
#include <algorithm>
#include <assert.h>
#include <cstring>

#include "error.h"
#include "tensor.h"
#include "attributes.h"

struct product_plan
{
    size_t copy_size_D;
    TAPP_datatype type_D;
    TAPP_element_op op_D;
    cutensorPlan_t* contraction_plan;
    cutensorPlan_t* permutation_plan;
};

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_ */
