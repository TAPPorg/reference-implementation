#ifndef TAPP_REF_IMPL_REF_IMPL_PERMUTE_H_
#define TAPP_REF_IMPL_REF_IMPL_PERMUTE_H_

#include <tapp/permute.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

struct plan
{
    // Permuted info
    int P_nmode;
    int64_t* P_idx;
    int64_t* P_extents;
    int64_t* P_strides_A;
    int64_t* P_strides_C;
    int64_t P_size;

    // Isolated indices X info (reduced)
    int I_nmode;
    int64_t* I_idx;
    int64_t* I_extents;
    int64_t* I_strides;
    int64_t I_size;

    // Broadcasted indices info (isolated Z)
    int B_nmode;
    int64_t* B_idx;
    int64_t* B_extents;
    int64_t* B_strides;
    int64_t B_size;

    TAPP_prectype prec;
    TAPP_datatype type_A;
    TAPP_datatype type_B;
    TAPP_datatype type_C;
    TAPP_element_op op_A;
    TAPP_element_op op_B;
    TAPP_element_op op_C;
};

#endif  /* TAPP_REF_IMPL_REF_IMPL_PERMUTE_H_ */