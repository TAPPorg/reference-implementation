#ifndef TAPP_REF_IMPL_MULTI_TAPP_HANDLE_H_
#define TAPP_REF_IMPL_MULTI_TAPP_HANDLE_H_

#include "tapp/handle.h"
#include "tapp/attributes.h"
#include "tapp/error.h"
#include "tapp/executor.h"
#include "tapp/product.h"
#include "tapp/status.h"
#include "tapp/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h> 

#define REFERENCE 0
#define TBLIS 1

extern char *paths[];

struct Multi_TAPP_handle
{
    uint64_t impl_id;
    TAPP_handle tapp_handle;
    void* lib_handle;
    TAPP_error (*TAPP_attr_set)(TAPP_attr attr, TAPP_handle handle, TAPP_key key, void* value);
    TAPP_error (*TAPP_attr_get)(TAPP_attr attr, TAPP_handle handle, TAPP_key key, void** value);
    TAPP_error (*TAPP_attr_clear)(TAPP_attr attr, TAPP_handle handle, TAPP_key key);
    bool (*TAPP_check_success)(TAPP_error error, TAPP_handle handle);
    size_t (*TAPP_explain_error)(TAPP_error error, TAPP_handle handle, size_t maxlen, char* message);
    TAPP_error (*TAPP_create_executor)(TAPP_executor* exec, TAPP_handle handle);
    TAPP_error (*TAPP_destroy_executor)(TAPP_executor exec, TAPP_handle handle);
    TAPP_error (*TAPP_create_handle)(uint64_t impl_id, TAPP_handle* handle);
    TAPP_error (*TAPP_destroy_handle)(TAPP_handle handle);
    TAPP_error (*TAPP_create_tensor_product)(TAPP_tensor_product* plan,
                                             TAPP_handle handle,
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
    TAPP_error (*TAPP_destroy_tensor_product)(TAPP_tensor_product plan, TAPP_handle handle);
    TAPP_error (*TAPP_execute_product)(TAPP_tensor_product plan,
                                       TAPP_handle handle,
                                       TAPP_executor exec,
                                       TAPP_status* status,
                                       const void* alpha,
                                       const void* A,
                                       const void* B,
                                       const void* beta,
                                       const void* C,
                                             void* D);
    TAPP_error (*TAPP_execute_batched_product)(TAPP_tensor_product plan,
                                               TAPP_handle handle,
                                               TAPP_executor exec,
                                               TAPP_status* status,
                                               int num_batches,
                                               const void* alpha,
                                               const void** A,
                                               const void** B,
                                               const void* beta,
                                               const void** C,
                                                     void** D);
    TAPP_error (*TAPP_destroy_status)(TAPP_status status, TAPP_handle handle);
    TAPP_error (*TAPP_create_tensor_info)(TAPP_tensor_info* info,
                                          TAPP_handle handle,
                                          TAPP_datatype type,
                                          int nmode,
                                          const int64_t* extents,
                                          const int64_t* strides);
    TAPP_error (*TAPP_destroy_tensor_info)(TAPP_tensor_info info, TAPP_handle handle);
    int (*TAPP_get_nmodes)(TAPP_tensor_info info, TAPP_handle handle);
    TAPP_error (*TAPP_set_nmodes)(TAPP_tensor_info info, TAPP_handle handle, int nmodes);
    void (*TAPP_get_extents)(TAPP_tensor_info info, TAPP_handle handle, int64_t* extents);
    TAPP_error (*TAPP_set_extents)(TAPP_tensor_info info, TAPP_handle handle, const int64_t* extents);
    void (*TAPP_get_strides)(TAPP_tensor_info info, TAPP_handle handle, int64_t* strides);
    TAPP_error (*TAPP_set_strides)(TAPP_tensor_info info, TAPP_handle handle, const int64_t* strides);
};

#endif /* TAPP_REF_IMPL_MULTI_TAPP_HANDLE_H_ */