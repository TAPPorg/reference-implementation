/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2024
 */
#include "ref_impl.h"
#include <string.h>


bool TAPP_check_success(TAPP_error error) {
    return error.type == TAPP_ERROR_TYPE_TAPP && error.code == TAPP_ERR_SUCCESS;
}


size_t TAPP_explain_error(TAPP_error error,
                          size_t maxlen,
                          char* message) {
    const char* error_message;
    switch (error.type)
    {
    case TAPP_ERROR_TYPE_TAPP:
        error_message = tapp_error_string(error.code);
        break;
    default:
        error_message = "Unknown error type.";
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