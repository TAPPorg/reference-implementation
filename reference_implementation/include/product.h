#ifndef TAPP_REF_IMPL_REF_IMPL_PRODUCT_H_
#define TAPP_REF_IMPL_REF_IMPL_PRODUCT_H_

#include <tapp/product.h>

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef TAPP_REFERENCE_ENABLE_TBLIS
#include "tblis_bind.h"
#endif

#include "tensor.h"

struct plan
{
    int H_nmode;
    int P_nmode;
    int FA_nmode;
    int FB_nmode;
    int IA_nmode;
    int IB_nmode;
    int64_t* H_idx;
    int64_t* P_idx;
    int64_t* FA_idx;
    int64_t* FB_idx;
    int64_t* IA_idx;
    int64_t* IB_idx;
    int64_t* H_extents;
    int64_t* P_extents;
    int64_t* FA_extents;
    int64_t* FB_extents;
    int64_t* IA_extents;
    int64_t* IB_extents;
    int64_t* H_strides_A;
    int64_t* H_strides_B;
    int64_t* H_strides_D;
    int64_t* P_strides_A;
    int64_t* P_strides_B;
    int64_t* FA_strides_A;
    int64_t* FA_strides_D;
    int64_t* FB_strides_B;
    int64_t* FB_strides_D;
    int64_t* IA_strides_A;
    int64_t* IB_strides_B;
    int64_t H_size;
    int64_t P_size;
    int64_t FA_size;
    int64_t FB_size;
    int64_t IA_size;
    int64_t IB_size;
    TAPP_prectype prec;
    TAPP_datatype type_A;
    TAPP_datatype type_B;
    TAPP_datatype type_C;
    TAPP_datatype type_D;
    TAPP_element_op op_A;
    TAPP_element_op op_B;
    TAPP_element_op op_C;
    TAPP_element_op op_D;

    // These values are only used for the tblis part. Could be removed when we have a proper tblis mapping
    TAPP_tensor_info A;
    int64_t* idx_A;
    TAPP_tensor_info B;
    int64_t* idx_B;
    TAPP_tensor_info C;
    int64_t* idx_C;
    TAPP_tensor_info D;
    int64_t* idx_D;
};

#endif  /* TAPP_REF_IMPL_REF_IMPL_PRODUCT_H_ */