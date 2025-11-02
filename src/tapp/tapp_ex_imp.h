/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */

#ifndef TAPP_TAPP_EX_IMP_H_
#define TAPP_TAPP_EX_IMP_H_

#include "../tapp.h"
#include <stdint.h>

struct tensor_info
{
    TAPP_datatype type;
    int nmode;
    int64_t* extents;
    int64_t* strides;
    int uuid_len;
    char* uuid;
};

struct plan
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

TAPP_EXPORT TAPP_error create_executor(TAPP_executor* exec);
TAPP_EXPORT TAPP_error create_handle(TAPP_handle* handle);


#endif  // TAPP_TAPP_EX_IMP_H_
