#include "internal.h"


bool TAPP_check_success(TAPP_error error) {
    return error == TAPP_SUCCESS;
}


size_t TAPP_explain_error(TAPP_error error,
                          size_t maxlen,
                          char* message) {

    std::string str;

    int base = ((int) error)/100000;
    int code = ((int) error)%100000;

    switch (base*100000)
    {
    case TAPP_ERROR_TYPE_TAPP_BASE:
        if(code>0) str = tapp_error_string(code);
        break;
    case TAPP_ERROR_TYPE_CUDA_BASE:
        str = std::string("[CUDA Error]: ") + cudaGetErrorString(static_cast<cudaError_t>(code)) + "\n";
        break;
    case TAPP_ERROR_TYPE_CUBLAS_BASE:
        str = std::string("[cuBLAS Status]: ") + cublasGetStatusString(static_cast<cublasStatus_t>(code)) + "\n";
        break;
    case TAPP_ERROR_TYPE_CUTT_BASE:
        str = std::string("[cuTT Status]: ") + std::to_string(code);
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
