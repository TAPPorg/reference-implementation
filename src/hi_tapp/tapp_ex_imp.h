/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "../hi_tapp.h"
#include <stdint.h>

struct tensor_info
{
    HI_TAPP_datatype type;
    int nmode;
    int64_t* extents;
    int64_t* strides;
};

struct plan
{
    HI_TAPP_handle handle;
    HI_TAPP_element_op op_A;
    HI_TAPP_tensor_info A;
    int64_t* idx_A;
    HI_TAPP_element_op op_B;
    HI_TAPP_tensor_info B;
    int64_t* idx_B;
    HI_TAPP_element_op op_C;
    HI_TAPP_tensor_info C;
    int64_t* idx_C;
    HI_TAPP_element_op op_D;
    HI_TAPP_tensor_info D;
    int64_t* idx_D;
    HI_TAPP_prectype prec;
};

HI_TAPP_error create_executor(HI_TAPP_executor* exec);
HI_TAPP_error create_handle(HI_TAPP_handle* handle);
