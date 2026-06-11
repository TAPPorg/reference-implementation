#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_PRODUCT_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_PRODUCT_H_

#include <tapp/product.h>

#include <cstdint>

#include "error.h"
#include "handle.h"
#include "tensor.h"
#include "attributes.h"
#include "datatype.h"

#include "../ttgt/ttgt_plan.h"
#include "../ttgt/ttgt_utils.h"
#include "../ttgt/ttgt_optimizer.h"

// A built tensor-contraction plan for the cuBLAS/TTGT back-end.
//
// The vendored TTGTPlan holds the transpose + GEMM schedule (mirrors what
// TAPP_create_tensor_product computes once and TAPP_execute_product replays).
// The ContractionInfo it is driven by stores raw pointers into the dim/mode/
// scalar arrays below, so this struct owns that backing storage for the
// lifetime of the plan.
struct product_plan
{
    TTGTPlan* ttgt_plan;
    ContractionInfo* info;

    // Backing storage referenced by *info.
    int* dimA;
    int* dimB;
    int* dimC;
    int32_t* modeA;
    int32_t* modeB;
    int32_t* modeC;

    // alpha/beta are not known until execute time; these buffers are filled in
    // TAPP_execute_product and pointed at by info->alpha / info->beta.
    // Sized to hold the largest supported scalar (complex double).
    unsigned char alpha_storage[16];
    unsigned char beta_storage[16];

    // Host<->device transfer description for each operand.
    size_t copy_size_A;
    int64_t data_offset_A;
    size_t copy_size_B;
    int64_t data_offset_B;
    size_t copy_size_C;
    int64_t data_offset_C;
    size_t copy_size_D;
    int64_t data_offset_D;

    TAPP_datatype type_D;
    TAPP_element_op op_D;

    TAPP_handle handle;
};

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_PRODUCT_H_ */
