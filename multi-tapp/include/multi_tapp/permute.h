#ifndef TAPP_MULTI_TAPP_PERMUTE_H_
#define TAPP_MULTI_TAPP_PERMUTE_H_

#include "backend.h"
#include "tapp/permute.h"

TAPP_EXPORT TAPP_error Multi_TAPP_create_tensor_permute(Multi_TAPP_backend backend,
                                                        TAPP_tensor_permute* plan,
                                                        TAPP_executor exec,
                                                        TAPP_element_op op_A,
                                                        TAPP_tensor_info A,
                                                        const int64_t* idx_A,
                                                        TAPP_element_op op_B,
                                                        TAPP_tensor_info B,
                                                        const int64_t* idx_B,
                                                        TAPP_element_op op_C,
                                                        TAPP_tensor_info C,
                                                        const int64_t* idx_C,
                                                        TAPP_prectype prec);

TAPP_EXPORT TAPP_error Multi_TAPP_destroy_tensor_permute(Multi_TAPP_backend backend, TAPP_tensor_permute plan);

TAPP_EXPORT TAPP_error Multi_TAPP_execute_permute(Multi_TAPP_backend backend,
                                                  TAPP_tensor_permute plan,
                                                  TAPP_executor exec,
                                                  TAPP_status* status,
                                                  const void* alpha,
                                                  const void* A,
                                                  const void* beta,
                                                  const void* B,
                                                        void* C);

#endif /* TAPP_MULTI_TAPP_PERMUTE_H_ */
