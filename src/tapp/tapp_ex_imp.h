/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */

#ifndef TAPP_TAPP_EX_IMP_H_
#define TAPP_TAPP_EX_IMP_H_

#include "../tapp.h"
#include "helpers.h"
#include <stdint.h>

struct tensor_info
{
    TAPP_datatype type;
    int nmode;
    int64_t* extents;
    int64_t* strides;
};

struct product_plan
{
    TAPP_handle handle;
    TAPP_element_op op_A;
    TAPP_tensor_info A;
    int64_t* idx_A;
    TAPP_element_op op_B;
    TAPP_tensor_info B;
    int64_t* idx_B;
    TAPP_element_op op_C;
    TAPP_tensor_info C;
    int64_t* idx_C;
    TAPP_element_op op_D;
    TAPP_tensor_info D;
    int64_t* idx_D;
    TAPP_prectype prec;
};

struct transpose_plan
{
    TAPP_handle handle;
    TAPP_element_op op_X;
    TAPP_tensor_info X;
    int64_t* idx_X;
    TAPP_element_op op_Y;
    TAPP_tensor_info Y;
    int64_t* idx_Y;
    TAPP_element_op op_Z;
    TAPP_tensor_info Z;
    int64_t* idx_Z;
};


TAPP_EXPORT TAPP_error create_executor(TAPP_executor* exec);
TAPP_EXPORT TAPP_error create_handle(TAPP_handle* handle);

#endif  // TAPP_TAPP_EX_IMP_H_
