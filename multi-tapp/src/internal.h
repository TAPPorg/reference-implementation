#ifndef TAPP_MULTI_TAPP_INTERNAL_H_
#define TAPP_MULTI_TAPP_INTERNAL_H_

/*
 * Internal definitions shared by the multi-tapp source files: the objects behind the handles of multi-tapp and
 * helpers for the error codes. Not part of the public interface of multi-tapp.
 */

#include "../include/multi_tapp/backend.h"
#include "../include/multi_tapp/error.h"
#include "tapp/permute.h"
#include <tapp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* A loaded TAPP implementation and its functions. A function is NULL if the implementation does not provide it */
struct mt_backend
{
    void* lib_handle;
    TAPP_error (*TAPP_start)(void);
    TAPP_error (*TAPP_shutdown)(void);
    TAPP_error (*TAPP_create_attr)(TAPP_attr* attr);
    TAPP_error (*TAPP_destroy_attr)(TAPP_attr attr);
    TAPP_error (*TAPP_attr_set)(TAPP_attr attr, TAPP_key key, size_t size, void* value);
    TAPP_error (*TAPP_attr_get)(TAPP_attr attr, TAPP_key key, size_t* size, void** value);
    TAPP_error (*TAPP_attr_clear)(TAPP_attr attr, TAPP_key key);
    bool (*TAPP_check_success)(TAPP_error error);
    size_t (*TAPP_explain_error)(TAPP_error error, size_t maxlen, char* message);
    TAPP_error (*TAPP_create_executor)(TAPP_executor* exec, TAPP_attr attr);
    TAPP_error (*TAPP_destroy_executor)(TAPP_executor exec);
    TAPP_error (*TAPP_wait_one)(TAPP_executor exec, TAPP_status status);
    TAPP_error (*TAPP_wait_all)(TAPP_executor exec);
    TAPP_error (*TAPP_status_get_error)(TAPP_status status);
    TAPP_error (*TAPP_destroy_status)(TAPP_status status);
    TAPP_error (*TAPP_create_tensor_product)(TAPP_tensor_product* plan,
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
                                             TAPP_element_op op_D,
                                             TAPP_tensor_info D,
                                             const int64_t* idx_D,
                                             TAPP_prectype prec);
    TAPP_error (*TAPP_destroy_tensor_product)(TAPP_tensor_product plan);
    TAPP_error (*TAPP_execute_product)(TAPP_tensor_product plan,
                                       TAPP_executor exec,
                                       TAPP_status* status,
                                       const void* alpha,
                                       const void* A,
                                       const void* B,
                                       const void* beta,
                                       const void* C,
                                             void* D);
    TAPP_error (*TAPP_create_tensor_permute)(TAPP_tensor_permute* plan,
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
    TAPP_error (*TAPP_destroy_tensor_permute)(TAPP_tensor_permute plan);
    TAPP_error (*TAPP_execute_permute)(TAPP_tensor_permute plan,
                                       TAPP_executor exec,
                                       TAPP_status* status,
                                       const void* alpha,
                                       const void* A,
                                       const void* beta,
                                       const void* B,
                                             void* C);
    TAPP_error (*TAPP_execute_batched_product)(TAPP_tensor_product plan,
                                               TAPP_executor exec,
                                               TAPP_status* status,
                                               int num_batches,
                                               const void* alpha,
                                               const void** A,
                                               const void** B,
                                               const void* beta,
                                               const void** C,
                                                     void** D);
    TAPP_error (*TAPP_create_tensor_info)(TAPP_tensor_info* info,
                                          TAPP_datatype type,
                                          int nmode,
                                          const int64_t* extents,
                                          const int64_t* strides);
    TAPP_error (*TAPP_destroy_tensor_info)(TAPP_tensor_info info);
    int (*TAPP_get_nmodes)(TAPP_tensor_info info);
    TAPP_error (*TAPP_set_nmodes)(TAPP_tensor_info info, int nmodes);
    void (*TAPP_get_extents)(TAPP_tensor_info info, int64_t* extents);
    TAPP_error (*TAPP_set_extents)(TAPP_tensor_info info, const int64_t* extents);
    void (*TAPP_get_strides)(TAPP_tensor_info info, int64_t* strides);
    TAPP_error (*TAPP_set_strides)(TAPP_tensor_info info, const int64_t* strides);
};

/* The objects multi-tapp hands out: the object of the backend together with the backend that created it */
struct mt_attr
{
    Multi_TAPP_backend backend;
    TAPP_attr attr;
};

struct mt_executor
{
    Multi_TAPP_backend backend;
    TAPP_executor exec;
};

struct mt_tensor_info
{
    Multi_TAPP_backend backend;
    TAPP_tensor_info info;
};

struct mt_tensor_product
{
    Multi_TAPP_backend backend;
    TAPP_tensor_product plan;
};

struct mt_tensor_permute
{
    Multi_TAPP_backend backend;
    TAPP_tensor_permute plan;
};

struct mt_status
{
    Multi_TAPP_backend backend;
    TAPP_status status;
};

/* Convert an error code returned by a backend into the error code multi-tapp returns */
static inline TAPP_error multi_tapp_error_from_backend(TAPP_error error)
{
    return error > 0 ? error + MULTI_TAPP_ERROR_LAST : error;
}

/* Check an error code returned directly by a backend, before it is converted */
static inline bool multi_tapp_check_success(TAPP_error error)
{
    return error == MULTI_TAPP_SUCCESS;
}

/*
 * The status of an execute function. Before the operation is submitted, multi_tapp_status_wrap allocates the wrapper
 * (none if status is NULL: the caller does not want one), so that a failed allocation cannot leave an operation running
 * that the caller has no status for. multi_tapp_status_of gives the pointer the backend writes its status to.
 * multi_tapp_status_confirm hands the wrapper to the caller if the backend succeeded and frees it otherwise, and returns
 * the converted error of the backend.
 */
static inline TAPP_error multi_tapp_status_wrap(Multi_TAPP_backend backend, TAPP_status* status, struct mt_status** wrapper)
{
    *wrapper = NULL;
    if (status == NULL) return MULTI_TAPP_SUCCESS;

    *wrapper = malloc(sizeof(struct mt_status));
    if (*wrapper == NULL)
    {
        fprintf(stderr, "ERROR: Failed to allocate space for a status object\n");
        return MULTI_TAPP_ERROR_ALLOCATION_FAILED;
    }
    (*wrapper)->backend = backend;
    (*wrapper)->status = 0;
    return MULTI_TAPP_SUCCESS;
}

static inline TAPP_status* multi_tapp_status_of(struct mt_status* wrapper)
{
    return wrapper == NULL ? NULL : &wrapper->status;
}

static inline TAPP_error multi_tapp_status_confirm(struct mt_status* wrapper, TAPP_status* status, TAPP_error error)
{
    if (!multi_tapp_check_success(error))
    {
        free(wrapper);
        return multi_tapp_error_from_backend(error);
    }
    if (status != NULL) *status = (TAPP_status)wrapper;
    return MULTI_TAPP_SUCCESS;
}

#endif /* TAPP_MULTI_TAPP_INTERNAL_H_ */
