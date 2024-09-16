/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "../tapp.h"
#include <stdint.h>

struct tensor_info
{
    TAPP_datatype type;
    int nmode;
    int64_t* extents;
    int64_t* strides;
};

struct plan
{
    TAPP_handle handle;

    int nmode_D;
    int contractions;
    int64_t size_D;
    int64_t size_contraction;

    int64_t* extents_D;
    int64_t* extents_contraction;
    int64_t* free_strides_A;
    int64_t* contracted_strides_A;
    int64_t* free_strides_B;
    int64_t* contracted_strides_B;

    int64_t* strides_C;
    int64_t* strides_D;

    TAPP_datatype type_A;
    TAPP_datatype type_B;
    TAPP_datatype type_C;
    TAPP_datatype type_D;

    TAPP_element_op op_A;
    TAPP_element_op op_B;
    TAPP_element_op op_C;
    TAPP_element_op op_D;
    TAPP_prectype prec;
};

