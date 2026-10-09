#ifndef TAPP_MULTI_TAPP_PRODUCT_H_
#define TAPP_MULTI_TAPP_PRODUCT_H_

#include "backend.h"
#include "tensor.h"
#include "executor.h"
#include "tapp/product.h"

struct Multi_TAPP_tensor_product
{
    TAPP_backend backend;
    TAPP_tensor_product plan;
};

TAPP_EXPORT TAPP_error Multi_TAPP_create_tensor_product(TAPP_backend backend,
                                                        TAPP_tensor_product* plan,
                                                        TAPP_element_op op_A,
                                                        TAPP_tensor_info A,
                                                        const int64_t* idx_A,
                                                        TAPP_element_op op_B,
                                                        TAPP_tensor_info B,
                                                        const int64_t* idx_B,
                                                        TAPP_element_op op_C,
                                                        TAPP_tensor_info C,
                                                        const int64_t* idx_C,
                                                        TAPP_element_op op_D,
                                                        TAPP_tensor_info D,
                                                        const int64_t* idx_D,
                                                        TAPP_prectype prec);

TAPP_EXPORT TAPP_error Multi_TAPP_destroy_tensor_product(TAPP_backend backend, TAPP_tensor_product plan);

TAPP_EXPORT TAPP_error Multi_TAPP_execute_product(TAPP_backend backend,
                                                  TAPP_tensor_product plan,
                                                  TAPP_executor exec,
                                                  TAPP_status* status,
                                                  const void* alpha,
                                                  const void* A,
                                                  const void* B,
                                                  const void* beta,
                                                  const void* C,
                                                        void* D);

TAPP_EXPORT TAPP_error Multi_TAPP_execute_batched_product(TAPP_backend backend,
                                                          TAPP_tensor_product plan,
                                                          TAPP_executor exec,
                                                          TAPP_status* status,
                                                          int num_batches,
                                                          const void* alpha,
                                                          const void** A,
                                                          const void** B,
                                                          const void* beta,
                                                          const void** C,
                                                                void** D);

#endif /* TAPP_MULTI_TAPP_PRODUCT_H_ */