#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_

#include <tapp/permute.h>

#include <cutensor.h>

#include <vector>
#include <algorithm>
#include <assert.h>
#include <cstring>

#include "error.h"
#include "tensor.h"
#include "attributes.h"

struct permute_plan
{
    cutensorPlan_t* main_permutation_plan;    // Perform alpha op(A) + beta op(B)
    cutensorPlan_t* result_permutation_plan; // Apply op_C
    TAPP_element_op op_C;
    TAPP_datatype type_C;
    size_t copy_size_C;
};

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_PRODUCT_H_ */
