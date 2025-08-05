#ifndef HI_TAPP_PRODUCT_H_
#define HI_TAPP_PRODUCT_H_

#include <stdint.h>

#include "hi_tapp/error.h"
#include "hi_tapp/handle.h"
#include "hi_tapp/executor.h"
#include "hi_tapp/datatype.h"
#include "hi_tapp/status.h"
#include "hi_tapp/tensor.h"

//TODO: where should this go?
typedef int HI_TAPP_element_op;

enum
{
    HI_TAPP_IDENTITY = 0,
    HI_TAPP_CONJUGATE = 1,
};

//TODO: where should this go?
#define HI_TAPP_IN_PLACE NULL

/*
 * TODO: what are the required error conditions?
 * TODO: must C and D info be the same? (should they just be the same variable?)
 */

typedef intptr_t HI_TAPP_tensor_product;

HI_TAPP_error HI_TAPP_create_tensor_product(HI_TAPP_tensor_product* plan,
                                      HI_TAPP_handle handle,
                                      HI_TAPP_element_op op_A,
                                      HI_TAPP_tensor_info A,
                                      const int64_t* idx_A,
                                      HI_TAPP_element_op op_B,
                                      HI_TAPP_tensor_info B,
                                      const int64_t* idx_B,
                                      HI_TAPP_element_op op_C,
                                      HI_TAPP_tensor_info C,
                                      const int64_t* idx_C,
                                      HI_TAPP_element_op op_D,
                                      HI_TAPP_tensor_info D,
                                      const int64_t* idx_D,
                                      HI_TAPP_prectype prec);

HI_TAPP_error HI_TAPP_destory_tensor_product(HI_TAPP_tensor_product plan);

//TODO: in-place operation: set C = NULL or HI_TAPP_IN_PLACE?

HI_TAPP_error HI_TAPP_execute_product(HI_TAPP_tensor_product plan,
                                HI_TAPP_executor exec,
                                HI_TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* B,
                                const void* beta,
                                const void* C,
                                      void* D);

//TODO: is it always OK to pass NULL for exec?
//TODO: can C be NULL/HI_TAPP_IN_PLACE (in addition to array entries being NULL)?

HI_TAPP_error HI_TAPP_execute_batched_product(HI_TAPP_tensor_product plan,
                                        HI_TAPP_executor exec,
                                        HI_TAPP_status* status,
                                        int num_batches,
                                        const void* alpha,
                                        const void** A,
                                        const void** B,
                                        const void* beta,
                                        const void** C,
                                              void** D);

#endif /* HI_TAPP_PRODUCT_H_ */
