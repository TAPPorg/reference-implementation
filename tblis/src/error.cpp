/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "internal.h"

#include <algorithm>
#include <cstring>

static const char* error_message(TAPP_error error)
{
    switch (static_cast<TAPP_TBLIS_error>(error))
    {
        case TAPP_TBLIS_SUCCESS:                    return "Success.";
        case TAPP_TBLIS_ERROR_NULL_POINTER:         return "A required pointer argument is null.";
        case TAPP_TBLIS_ERROR_NEGATIVE_NMODE:       return "The number of modes cannot be negative.";
        case TAPP_TBLIS_ERROR_NEGATIVE_EXTENT:      return "Extents cannot be negative.";
        case TAPP_TBLIS_ERROR_UNSUPPORTED_DATATYPE: return "The datatype is not supported by TBLIS.";
        case TAPP_TBLIS_ERROR_OUT_OF_MEMORY:        return "Memory allocation failed.";
        case TAPP_TBLIS_ERROR_DATATYPE_MISMATCH:    return "All tensors of an operation must have the same datatype.";
        case TAPP_TBLIS_ERROR_EXTENT_MISMATCH:      return "An index has different extents in different places.";
        case TAPP_TBLIS_ERROR_C_D_MISMATCH:         return "Tensors C and D must have the same indices and extents.";
        case TAPP_TBLIS_ERROR_B_C_MISMATCH:         return "Tensors B and C must have the same indices and extents.";
        case TAPP_TBLIS_ERROR_INDEX_NOT_IN_INPUTS:  return "The output tensor has an index that appears in none of the inputs.";
        case TAPP_TBLIS_ERROR_TOO_MANY_INDICES:     return "The product has more distinct indices than TBLIS supports.";
        case TAPP_TBLIS_ERROR_UNSUPPORTED_CONTRACTION:
            return "An index appearing only in A or only in B is not supported yet.";
        case TAPP_TBLIS_ERROR_UNSUPPORTED_ELEMENT_OP:
            return "The element operation is not supported.";
        case TAPP_TBLIS_ERROR_UNSUPPORTED_PRECISION:
            return "The requested precision is not supported.";
        case TAPP_TBLIS_ERROR_UNKNOWN_ATTR_KEY:
            return "The attribute key is not known to the TBLIS implementation.";
        case TAPP_TBLIS_ERROR_INVALID_ATTR_VALUE:
            return "The value is not valid for this attribute key.";
        case TAPP_TBLIS_ERROR_CPU_MASK_UNAVAILABLE:
            return "The executor cannot run on the CPUs of its CPU mask.";
        case TAPP_TBLIS_ERROR_INVALID_STATUS:
            return "The status does not belong to a product of this executor.";
        case TAPP_TBLIS_ERROR_REPEATED_INDEX:
            return "A tensor of a permute uses an index more than once.";
        case TAPP_TBLIS_ERROR_INDEX_NOT_IN_OUTPUT:
            return "Tensor A of a permute has an index that does not appear in C.";
        case TAPP_TBLIS_ERROR_STATUS_WRONG_EXECUTOR:
            return "The status belongs to an operation of another executor.";
        case TAPP_TBLIS_STATUS_RUNNING:
            return "The operation of the status has not finished yet.";
        case TAPP_TBLIS_ERROR_INTERNAL:
            return "The operation failed with an unexpected internal error.";
    }

    return "Unknown error code.";
}

bool TAPP_check_success(TAPP_error error) {
    return error == TAPP_TBLIS_SUCCESS;
}

size_t TAPP_explain_error(TAPP_error error,
                          size_t maxlen,
                          char* message) {
    const char* msg = error_message(error);
    size_t len = std::strlen(msg);

    if (maxlen == 0) return len;

    size_t written = std::min(len, maxlen - 1);
    std::memcpy(message, msg, written);
    message[written] = '\0';

    return written;
}
