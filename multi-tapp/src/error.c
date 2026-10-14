#include "../include/multi_tapp/error.h"
#include "internal.h"

static bool is_multi_tapp_error(TAPP_error error);
static TAPP_error multi_tapp_error_to_backend(TAPP_error error);
static size_t write_error_message(char* message, size_t maxlen, const char* error_message);

bool Multi_TAPP_check_success(Multi_TAPP_backend backend,
                              TAPP_error error)
{
    // Errors of multi-tapp itself, the backend is not needed (and might not exist)
    if (is_multi_tapp_error(error)) return false;

    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return false;
    }

    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_check_success == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_check_success\n");
        return false; // TODO: Return error for non implemented function
    }
    return backend_ptr->TAPP_check_success(multi_tapp_error_to_backend(error));
}

size_t Multi_TAPP_explain_error(Multi_TAPP_backend backend,
                                TAPP_error error,
                                size_t maxlen,
                                char* message)
{
    // Errors of multi-tapp itself, the backend is not needed (and might not exist)
    switch (error)
    {
    case MULTI_TAPP_ERROR_DLOPEN_FAILED:
        return write_error_message(message, maxlen,"dlopen failed");
    case MULTI_TAPP_ERROR_MISSING_CREATE_TENSOR_PRODUCT:
        return write_error_message(message, maxlen,"dlsym failed to load crucial function TAPP_create_tensor_product");
    case MULTI_TAPP_ERROR_MISSING_EXECUTE_PRODUCT:
        return write_error_message(message, maxlen,"dlsym failed to load crucial function TAPP_execute_product");
    case MULTI_TAPP_ERROR_MISSING_CREATE_TENSOR_INFO:
        return write_error_message(message, maxlen,"dlsym failed to load crucial function TAPP_create_tensor_info");
    case MULTI_TAPP_ERROR_MISSING_CREATE_EXECUTOR:
        return write_error_message(message, maxlen,"dlsym failed to load crucial function TAPP_create_executor");
    case MULTI_TAPP_ERROR_MISSING_DESTROY_EXECUTOR:
        return write_error_message(message, maxlen,"dlsym failed to load crucial function TAPP_destroy_executor");
    case MULTI_TAPP_ERROR_MISSING_DESTROY_TENSOR_PRODUCT:
        return write_error_message(message, maxlen,"dlsym failed to load crucial function TAPP_destroy_tensor_product");
    case MULTI_TAPP_ERROR_MISSING_DESTROY_TENSOR_INFO:
        return write_error_message(message, maxlen,"dlsym failed to load crucial function TAPP_destroy_tensor_info");
    case MULTI_TAPP_ERROR_UNIMPLEMENTED:
        return write_error_message(message, maxlen,"Attempted to call an unloaded function, might not be implemented");
    case MULTI_TAPP_ERROR_ATTR_WRONG_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of TAPP_attr from different implementation");
    case MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of TAPP_executor from different implementation");
    case MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of TAPP_tensor_info from different implementation");
    case MULTI_TAPP_ERROR_TENSOR_INFO_A_WRONG_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of TAPP_tensor_info A from different implementation");
    case MULTI_TAPP_ERROR_TENSOR_INFO_B_WRONG_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of TAPP_tensor_info B from different implementation");
    case MULTI_TAPP_ERROR_TENSOR_INFO_C_WRONG_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of TAPP_tensor_info C from different implementation");
    case MULTI_TAPP_ERROR_TENSOR_INFO_D_WRONG_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of TAPP_tensor_info D from different implementation");
    case MULTI_TAPP_ERROR_TENSOR_PRODUCT_WRONG_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of TAPP_tensor_product from different implementation");
    case MULTI_TAPP_ERROR_INVALID_BACKEND:
        return write_error_message(message, maxlen,"Attempted use of an invalid Multi_TAPP_backend");
    case MULTI_TAPP_ERROR_INVALID_ATTR:
        return write_error_message(message, maxlen,"Attempted use of an invalid TAPP_attr");
    case MULTI_TAPP_ERROR_INVALID_EXECUTOR:
        return write_error_message(message, maxlen,"Attempted use of an invalid TAPP_executor");
    case MULTI_TAPP_ERROR_INVALID_TENSOR_INFO:
        return write_error_message(message, maxlen,"Attempted use of an invalid TAPP_tensor_info");
    case MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_A:
        return write_error_message(message, maxlen,"Attempted use of an invalid TAPP_tensor_info A");
    case MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_B:
        return write_error_message(message, maxlen,"Attempted use of an invalid TAPP_tensor_info B");
    case MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_C:
        return write_error_message(message, maxlen,"Attempted use of an invalid TAPP_tensor_info C");
    case MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_D:
        return write_error_message(message, maxlen,"Attempted use of an invalid TAPP_tensor_info D");
    case MULTI_TAPP_ERROR_INVALID_TENSOR_PRODUCT:
        return write_error_message(message, maxlen,"Attempted use of an invalid TAPP_tensor_product");
    case MULTI_TAPP_ERROR_INVALID_ARGUMENT:
        return write_error_message(message, maxlen,"An argument for the result of the function is NULL");
    case MULTI_TAPP_ERROR_ALLOCATION_FAILED:
        return write_error_message(message, maxlen,"Failed to allocate memory for a multi-tapp object");
    default:
        break;
    }
    if (is_multi_tapp_error(error))
    {
        return write_error_message(message, maxlen,"Unknown multi-tapp error");
    }

    // Only the backend can explain its own error codes
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return write_error_message(message, maxlen,"No backend given to explain an error code of a backend");
    }

    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_explain_error == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_explain_error\n");
        return write_error_message(message, maxlen,"TAPP_explain_error is not implemented");
    }
    return backend_ptr->TAPP_explain_error(multi_tapp_error_to_backend(error), maxlen, message);
}

/* Check if an error code returned by multi-tapp is one of its own, and not an error code from a backend */
static bool is_multi_tapp_error(TAPP_error error)
{
    return error > MULTI_TAPP_SUCCESS && error <= MULTI_TAPP_ERROR_LAST;
}

/* Convert an error code returned by multi-tapp, that is not one of its own, back into the error code of the backend */
static TAPP_error multi_tapp_error_to_backend(TAPP_error error)
{
    return error > MULTI_TAPP_ERROR_LAST ? error - MULTI_TAPP_ERROR_LAST : error;
}

static size_t write_error_message(char* message, size_t maxlen, const char* error_message)
{
    size_t error_message_len = strlen(error_message);
    if (maxlen == 0) return error_message_len;

    if (message == NULL)
    {
        fprintf(stderr, "ERROR: Tried to write error message to NULL buffer\n");
        return 0;
    }

    size_t write_len = maxlen - 1 < error_message_len ? maxlen - 1 : error_message_len;
    memcpy(message, error_message, write_len * sizeof(char));
    message[write_len] = '\0';
    return write_len;
}