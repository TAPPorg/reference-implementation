#ifndef TAPP_REF_IMPL_INTERNAL_H_
#define TAPP_REF_IMPL_INTERNAL_H_

/*
 * Internal definitions shared by the source files of the reference implementation of TAPP: the objects behind the
 * handles and the helpers of the product. Not part of the public interface, that is tapp.h.
 */

#include <tapp.h>

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------------------------------------------------
 * Tensors
 * ------------------------------------------------------------------------------------------------------------------ */

struct tensor_info
{
    TAPP_datatype type;
    int nmode;
    int64_t* extents;
    int64_t* strides;
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Products
 * ------------------------------------------------------------------------------------------------------------------ */

struct plan
{
    // Hadamard info
    int H_nmode;
    int64_t* H_idx;
    int64_t* H_extents;
    int64_t* H_strides_A;
    int64_t* H_strides_B;
    int64_t* H_strides_D;
    int64_t H_size;

    // Contraction info
    int P_nmode;
    int64_t* P_idx;
    int64_t* P_extents;
    int64_t* P_strides_A;
    int64_t* P_strides_B;
    int64_t P_size;

    // Free indices A info
    int FA_nmode;
    int64_t* FA_idx;
    int64_t* FA_extents;
    int64_t* FA_strides_A;
    int64_t* FA_strides_D;
    int64_t FA_size;

    // Free indices B info
    int FB_nmode;
    int64_t* FB_idx;
    int64_t* FB_extents;
    int64_t* FB_strides_B;
    int64_t* FB_strides_D;
    int64_t FB_size;

    // Isolated indices A info (reduced)
    int IA_nmode;
    int64_t* IA_idx;
    int64_t* IA_extents;
    int64_t* IA_strides_A;
    int64_t IA_size;

    // Isolated indices B info (reduced)
    int IB_nmode;
    int64_t* IB_idx;
    int64_t* IB_extents;
    int64_t* IB_strides_B;
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
};

// Helpers of product_template.c, which is compiled once per combination of types (see product_instantiations.gen.c)
int64_t calcualte_offset(int64_t* coords, int nmode, int64_t* strides);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
int check_executor_existence(TAPP_executor exec, int error_code);

#endif  /* TAPP_REF_IMPL_INTERNAL_H_ */
