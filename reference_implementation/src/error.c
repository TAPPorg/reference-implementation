/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2024
 */
#include <ref_imp.h>
#include <string.h>


bool TAPP_check_success(TAPP_error error) {
    return error == 0;
}


size_t TAPP_explain_error(TAPP_error error,
                          size_t maxlen,
                          char* message) {
    char* error_message;
    switch (error)
    {
    case 0:
        error_message = "Success.";
        break;
    case 1:
        error_message = "The extents for the indices shared between tensor A and B does not match.";
        break;
    case 2:
        error_message = "The extents for the indices shared between tensor A and D does not match.";
        break;
    case 3:
        error_message = "The extents for the indices shared between tensor B and D does not match.";
        break;
    case 4:
        error_message = "Tensor D has indices not shared with tensor A or B.";
        break;
    case 5:
        error_message = "The tensors C and D have different amount of dimensions.";
        break;
    case 6:
        error_message = "The indices of tensor C and D does not line up.";
        break;
    case 7:
        error_message = "The extents for the indices shared between tensor C and D does not match.";
        break;
    case 8:
        error_message = "Aliasing found within tensor D.";
        break;
    case 9:
        error_message = "An idx in tensor A has two different extents.";
        break;
    case 10:
        error_message = "An idx in tensor B has two different extents.";
        break;
    case 11:
        error_message = "An idx in tensor D has two different extents.";
        break;
    case 12:
        error_message = "C should not be NULL while beta is not zero.";
        break;
    case 13:
        error_message = "Nmode can not be negative.";
        break;
    case 14:
        error_message = "Extents can not be negative.";
        break;
    default:
        break;
    }
    size_t message_len = strlen(error_message);
    if (maxlen == 0) {
        return message_len;
    }
    size_t writelen = maxlen - 1 < message_len ? maxlen - 1 : message_len;
    strncpy(message, error_message, writelen);
    message[writelen] = '\0';
    return writelen;
}