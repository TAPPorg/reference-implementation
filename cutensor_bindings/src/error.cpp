#include "../include/error.h"


bool TAPP_check_success(TAPP_error error) {
    return error.type == TAPP_ERROR_TYPE_TAPP && error.code == TAPP_ERR_SUCCESS;
}


size_t TAPP_explain_error(TAPP_error error,
                          size_t maxlen,
                          char* message) {

    std::string str;

    switch (error.type)
    {
    case TAPP_ERROR_TYPE_TAPP:
        str = tapp_error_string(error.code);
        break;
    case TAPP_ERROR_TYPE_CUDA:
        str = std::string("[CUDA Error]: ") + cudaGetErrorString(static_cast<cudaError_t>(error.code));
        break;
    case TAPP_ERROR_TYPE_CUTENSOR:
        str = std::string("[cuTENSOR Status]: ") + cutensorGetErrorString(static_cast<cutensorStatus_t>(error.code));
        break;
    default:
        str = "Unknown error type.";
        break;
    }

    const char* error_message = str.c_str();
    size_t message_len = strlen(error_message);
    if (maxlen == 0) {
        return message_len;
    }
    size_t writelen = maxlen - 1 < message_len ? maxlen - 1 : message_len;
    strncpy(message, error_message, writelen);
    message[writelen] = '\0';
    return writelen;
}
