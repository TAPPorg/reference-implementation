/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - November 2025
 */

#ifndef TAPP_TRANSPOSE_H_
#define TAPP_TRANSPOSE_H_

#include <stdint.h>

#include "util.h"
#include "error.h"
#include "handle.h"
#include "executor.h"
#include "datatype.h"
#include "status.h"
#include "tensor.h"

typedef intptr_t TAPP_tensor_transpose;

TAPP_EXPORT TAPP_error TAPP_create_tensor_transpose(TAPP_tensor_transpose* plan,
                                                    TAPP_handle handle,
                                                    TAPP_element_op op_X,
                                                    TAPP_tensor_info X,
                                                    const int64_t* idx_X,
                                                    TAPP_element_op op_Y,
                                                    TAPP_tensor_info Y,
                                                    const int64_t* idx_Y,
                                                    TAPP_element_op op_Z,
                                                    TAPP_tensor_info Z,
                                                    const int64_t* idx_Z);

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_transpose(TAPP_tensor_transpose plan);
 
TAPP_EXPORT TAPP_error TAPP_execute_transpose(TAPP_tensor_transpose plan,
                                              TAPP_executor exec,
                                              TAPP_status* status,
                                              const void* alpha,
                                              const void* X,
                                              const void* Y,
                                              const void* beta,
                                                    void* Z);
#endif /* TAPP_TRANSPOSE_H_ */
