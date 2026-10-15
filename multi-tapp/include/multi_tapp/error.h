#ifndef TAPP_MULTI_TAPP_ERROR_H_
#define TAPP_MULTI_TAPP_ERROR_H_

#include "backend.h"
#include "tapp/error.h"

/*
 * Error codes:
 *
 * Zero indicates success. The values 1 to MULTI_TAPP_ERROR_LAST are the errors of multi-tapp itself. A positive
 * error code from a backend is returned with MULTI_TAPP_ERROR_LAST added to it, so it can not be mistaken for an
 * error of multi-tapp. Multi_TAPP_check_success and Multi_TAPP_explain_error remove the offset again before they
 * ask the backend. Other values from a backend (zero and negative) are returned unchanged.
 *
 * The offset changes whenever multi-tapp gets a new error code, so do not rely on the numerical values.
 */
enum
{
    /* Success*/
    MULTI_TAPP_SUCCESS = 0,

    /* Loading a backend */
    MULTI_TAPP_ERROR_DLOPEN_FAILED,
    MULTI_TAPP_ERROR_MISSING_CREATE_TENSOR_PRODUCT,
    MULTI_TAPP_ERROR_MISSING_EXECUTE_PRODUCT,
    MULTI_TAPP_ERROR_MISSING_CREATE_TENSOR_INFO,
    MULTI_TAPP_ERROR_MISSING_CREATE_EXECUTOR,
    MULTI_TAPP_ERROR_MISSING_DESTROY_EXECUTOR,
    MULTI_TAPP_ERROR_MISSING_DESTROY_TENSOR_PRODUCT,
    MULTI_TAPP_ERROR_MISSING_DESTROY_TENSOR_INFO,

    /* The backend does not provide the called function */
    MULTI_TAPP_ERROR_UNIMPLEMENTED,

    /* An object was created by another backend than the one given in the call */
    MULTI_TAPP_ERROR_ATTR_WRONG_BACKEND,
    MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND,
    MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND,
    MULTI_TAPP_ERROR_TENSOR_INFO_A_WRONG_BACKEND,
    MULTI_TAPP_ERROR_TENSOR_INFO_B_WRONG_BACKEND,
    MULTI_TAPP_ERROR_TENSOR_INFO_C_WRONG_BACKEND,
    MULTI_TAPP_ERROR_TENSOR_INFO_D_WRONG_BACKEND,
    MULTI_TAPP_ERROR_TENSOR_PRODUCT_WRONG_BACKEND,
    MULTI_TAPP_ERROR_TENSOR_PERMUTE_WRONG_BACKEND,
    MULTI_TAPP_ERROR_STATUS_WRONG_BACKEND,

    /* A backend or an object given in the call is not valid (it is zero) */
    MULTI_TAPP_ERROR_INVALID_BACKEND,
    MULTI_TAPP_ERROR_INVALID_ATTR,
    MULTI_TAPP_ERROR_INVALID_EXECUTOR,
    MULTI_TAPP_ERROR_INVALID_TENSOR_INFO,
    MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_A,
    MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_B,
    MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_C,
    MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_D,
    MULTI_TAPP_ERROR_INVALID_TENSOR_PRODUCT,
    MULTI_TAPP_ERROR_INVALID_TENSOR_PERMUTE,
    MULTI_TAPP_ERROR_INVALID_STATUS,

    /* A pointer argument that multi-tapp writes its result to is NULL */
    MULTI_TAPP_ERROR_INVALID_ARGUMENT,

    /* multi-tapp could not allocate memory for one of its own objects */
    MULTI_TAPP_ERROR_ALLOCATION_FAILED,

    /* Not an error that is returned, only marks the end. New error codes must be added above it */
    MULTI_TAPP_ERROR_LAST,
};

TAPP_EXPORT bool Multi_TAPP_check_success(Multi_TAPP_backend backend, TAPP_error error);

TAPP_EXPORT size_t Multi_TAPP_explain_error(Multi_TAPP_backend backend,
                                            TAPP_error error,
                                            size_t maxlen,
                                            char* message);

#endif /* TAPP_MULTI_TAPP_ERROR_H_ */