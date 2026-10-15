#ifndef TAPP_PERMUTE_H_
#define TAPP_PERMUTE_H_

#include <stdint.h>

#include "util.h"
#include "error.h"
#include "executor.h"
#include "datatype.h"
#include "status.h"
#include "tensor.h"
#include "element_op.h"

typedef intptr_t TAPP_tensor_permute;

TAPP_EXPORT TAPP_error TAPP_create_tensor_permute(TAPP_tensor_permute* plan,
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

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_permute(TAPP_tensor_permute plan);
 
TAPP_EXPORT TAPP_error TAPP_execute_permute(TAPP_tensor_permute plan,
                                            TAPP_executor exec,
                                            TAPP_status* status,
                                            const void* alpha,
                                            const void* A,
                                            const void* beta,
                                            const void* B,
                                                  void* C);
#endif /* TAPP_PERMUTE_H_ */