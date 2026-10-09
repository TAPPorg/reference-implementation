/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "../include/error.h"

#include <algorithm>
#include <cstring>

static const char* error_message(TAPP_error error)
{
    switch (static_cast<error_code>(error))
    {
        case SUCCESS:                    return "Success.";
        case ERROR_NULL_POINTER:         return "A required pointer argument is null.";
        case ERROR_NEGATIVE_NMODE:       return "The number of modes cannot be negative.";
        case ERROR_NEGATIVE_EXTENT:      return "Extents cannot be negative.";
        case ERROR_UNSUPPORTED_DATATYPE: return "The datatype is not supported by TBLIS.";
        case ERROR_OUT_OF_MEMORY:        return "Memory allocation failed.";
        case ERROR_DATATYPE_MISMATCH:    return "All tensors in a product must have the same datatype.";
        case ERROR_EXTENT_MISMATCH:      return "An index has different extents in different places.";
        case ERROR_C_D_MISMATCH:         return "Tensors C and D must have the same indices and extents.";
        case ERROR_INDEX_NOT_IN_INPUTS:  return "Tensor D has an index that appears in neither A nor B.";
        case ERROR_TOO_MANY_INDICES:     return "The product has more distinct indices than TBLIS supports.";
        case ERROR_UNSUPPORTED_CONTRACTION:
            return "An index appearing only in A or only in B is not supported yet.";
        case ERROR_UNSUPPORTED_ELEMENT_OP:
            return "The element operation is not supported.";
        case ERROR_UNSUPPORTED_PRECISION:
            return "The requested precision is not supported.";
    }

    return "Unknown error code.";
}

bool TAPP_check_success(TAPP_error error) {
    return error == SUCCESS;
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
