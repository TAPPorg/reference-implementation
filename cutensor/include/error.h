#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_ERROR_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_ERROR_H_

#include <tapp/error.h>

#include <cutensor.h>

#include <cstring>
#include <string>

enum
{
    TAPP_SUCCESS          = 0,
    TAPP_ERR_EXTENTS_AB       = 1,
    TAPP_ERR_EXTENTS_AD       = 2,
    TAPP_ERR_EXTENTS_BD       = 3,
    TAPP_ERR_D_UNSHARED_IDX   = 4,
    TAPP_ERR_CD_NDIM          = 5,
    TAPP_ERR_CD_IDX           = 6,
    TAPP_ERR_EXTENTS_CD       = 7,
    TAPP_ERR_D_ALIASING       = 8,
    TAPP_ERR_A_IDX_EXTENTS    = 9,
    TAPP_ERR_B_IDX_EXTENTS    = 10,
    TAPP_ERR_D_IDX_EXTENTS    = 11,
    TAPP_ERR_C_NULL_BETA      = 12,
    TAPP_ERR_NMODE_NEGATIVE   = 13,
    TAPP_ERR_EXTENTS_NEGATIVE = 14,
    TAPP_ERR_INVALID_KEY      = 15,
    TAPP_ERR_NO_EXECUTOR      = 16,
    TAPP_ERR_NOT_IMPLEMENTED  = 17,
    TAPP_ERR_ATTR_CREATION_FAILED  = 18,
    TAPP_ERR_NULL_POINTER  = 19,
    TAPP_ERR_ATTR_INVALID_VALUE  = 20,
    TAPP_ERR_ATTR_UNKNOWN_KEY = 21,
    TAPP_ERR_ATTR_INVALID_STREAM = 22,
};


typedef enum
{
    TAPP_ERROR_TYPE_TAPP_BASE  = 0,
    TAPP_ERROR_TYPE_CUDA_BASE  = 100,
    TAPP_ERROR_TYPE_CUTENSOR_BASE = 200,
} TAPP_error_base;

inline TAPP_error tapp_error(cudaError_t e)      { return e == cudaSuccess ? TAPP_SUCCESS : (int) e+TAPP_ERROR_TYPE_CUDA_BASE; }
inline TAPP_error tapp_error(cutensorStatus_t e) { return e == CUTENSOR_STATUS_SUCCESS ? TAPP_SUCCESS : (int) e + TAPP_ERROR_TYPE_CUTENSOR_BASE; }

/* Message for a TAPP_ERROR_TYPE_TAPP code, shared so every implementation
 * reports these errors identically. */
static inline const char* tapp_error_string(int code)
{
    switch (code)
    {
    case TAPP_ERR_INVALID_KEY:      return "Invalid attribute key.";
    default:                        return "Unknown TAPP error code.";
    }
}

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDS_ERROR_H_ */
