#ifndef TAPP_ERROR_H_
#define TAPP_ERROR_H_

#include <stddef.h>
#include <stdbool.h>

#include "util.h"

/* The source an error code originates from. Type TAPP is value 0 so that a
 * fully-zero TAPP_error denotes success. */
typedef enum
{
    TAPP_ERROR_TYPE_TAPP     = 0,
    TAPP_ERROR_TYPE_CUDA     = 1,
    TAPP_ERROR_TYPE_CUTENSOR = 2,
    TAPP_ERROR_TYPE_CUBLAS   = 3,
    TAPP_ERROR_TYPE_TBLIS    = 4,
} TAPP_error_type;

/* TAPP's own basic error codes, shared across implementations. */
enum
{
    TAPP_ERR_SUCCESS          = 0,
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
    TAPP_ERR_UNSUPPORTED_DATATYPE = 18,
};

/* An error carries the source (type) and that source's own error code. Only the
 * first error that occurs is reported; errors from different sources are not combined. */
typedef struct
{
    int type;
    int code;
} TAPP_error;

static inline TAPP_error tapp_error(int type, int code)
{
    TAPP_error error;
    error.type = type;
    error.code = code;
    return error;
}

#define TAPP_SUCCESS tapp_error(TAPP_ERROR_TYPE_TAPP, TAPP_ERR_SUCCESS)

/* Message for a TAPP_ERROR_TYPE_TAPP code, shared so every implementation
 * reports these errors identically. */
static inline const char* tapp_error_string(int code)
{
    switch (code)
    {
    case TAPP_ERR_SUCCESS:          return "Success.";
    case TAPP_ERR_EXTENTS_AB:       return "The extents for the indices shared between tensor A and B does not match.";
    case TAPP_ERR_EXTENTS_AD:       return "The extents for the indices shared between tensor A and D does not match.";
    case TAPP_ERR_EXTENTS_BD:       return "The extents for the indices shared between tensor B and D does not match.";
    case TAPP_ERR_D_UNSHARED_IDX:   return "Tensor D has indices not shared with tensor A or B.";
    case TAPP_ERR_CD_NDIM:          return "The tensors C and D have different amount of dimensions.";
    case TAPP_ERR_CD_IDX:           return "The indices of tensor C and D does not line up.";
    case TAPP_ERR_EXTENTS_CD:       return "The extents for the indices shared between tensor C and D does not match.";
    case TAPP_ERR_D_ALIASING:       return "Aliasing found within tensor D.";
    case TAPP_ERR_A_IDX_EXTENTS:    return "An idx in tensor A has two different extents.";
    case TAPP_ERR_B_IDX_EXTENTS:    return "An idx in tensor B has two different extents.";
    case TAPP_ERR_D_IDX_EXTENTS:    return "An idx in tensor D has two different extents.";
    case TAPP_ERR_C_NULL_BETA:      return "C should not be NULL while beta is not zero.";
    case TAPP_ERR_NMODE_NEGATIVE:   return "Nmode can not be negative.";
    case TAPP_ERR_EXTENTS_NEGATIVE: return "Extents can not be negative.";
    case TAPP_ERR_INVALID_KEY:      return "Invalid attribute key.";
    case TAPP_ERR_NO_EXECUTOR:      return "Executor does not exist.";
    case TAPP_ERR_NOT_IMPLEMENTED:  return "Operation not implemented.";
    case TAPP_ERR_UNSUPPORTED_DATATYPE: return "Unsupported datatype for the back-end.";
    default:                        return "Unknown TAPP error code.";
    }
}

/* Return true if the error code indicates success and false otherwise. */
TAPP_EXPORT bool TAPP_check_success(TAPP_error error);

/*
 * Fill a user-provided buffer with an implementation-defined string explaining the error code. No more than maxlen-1
 * characters will be written. If maxlen is greater than zero, then a terminating null character is also
 * written. The actual number of characters written is returned, not including the terminating null character.
 * If maxlen is zero, then no characters are written and instead the length of the full string which would have been
 * written is returned, not including the terminating null character. This means that the message written will always
 * be null-terminated.
 *
 * TODO: should the null character be included in the return value?
 */
TAPP_EXPORT size_t TAPP_explain_error(TAPP_error error,
                                      size_t maxlen,
                                      char* message);

#endif /* TAPP_ERROR_H_ */
