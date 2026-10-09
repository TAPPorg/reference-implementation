#include "../include/error.h"

static size_t handle_error(char* message, size_t maxlen, char* error_message);

bool Multi_TAPP_check_success(TAPP_backend backend,
                              TAPP_error error)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_check_success == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_check_success\n");
        return false; // TODO: Return error for non implemented function
    }
    return error == 0;
    return backend_ptr->TAPP_check_success(error);
}

size_t Multi_TAPP_explain_error(TAPP_backend backend,
                                TAPP_error error,
                                size_t maxlen,
                                char* message)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_explain_error == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_explain_error\n");
        return handle_error(message, maxlen, "TAPP_explain_error is not implemented");
    }
    switch (error)
    {
    case 1:
        return handle_error(message, maxlen, "dlopen failed");
        break;
    case 2:
        return handle_error(message, maxlen, "dlsym failed to load crucial function TAPP_create_handle");
        break;
    case 3:
        return handle_error(message, maxlen, "dlsym failed to load crucial function TAPP_create_tensor_product");
        break;
    case 4:
        return handle_error(message, maxlen, "dlsym failed to load crucial function TAPP_execute_product");
        break;
    case 5:
        return handle_error(message, maxlen, "dlsym failed to load crucial function TAPP_create_tensor_info");
        break;
    case 6:
        return handle_error(message, maxlen, "Attempted to call an unloaded function, might not be implemented");
        break;
    case 7:
        return handle_error(message, maxlen, "Attempted use of TAPP_tensor_info from different implementation");
        break;
    case 8:
        return handle_error(message, maxlen, "Attempted use of TAPP_executor from different implementation");
        break;
    case 9:
        return handle_error(message, maxlen, "Attempted use of TAPP_tensor_info A from different implementation");
        break;
    case 10:
        return handle_error(message, maxlen, "Attempted use of TAPP_tensor_info B from different implementation");
        break;
    case 11:
        return handle_error(message, maxlen, "Attempted use of TAPP_tensor_info C from different implementation");
        break;
    case 12:
        return handle_error(message, maxlen, "Attempted use of TAPP_tensor_info D from different implementation");
        break;
    case 13:
        return handle_error(message, maxlen, "Attempted use of TAPP_tensor_product from different implementation");
        break;
    case 14:
        return handle_error(message, maxlen, "Attempted use of TAPP_status from different implementation");
        break;
    default:
        return backend_ptr->TAPP_explain_error(error, maxlen, message);
        break;
    }
}

static size_t handle_error(char* message, size_t maxlen, char* error_message)
{
    size_t error_message_len = strlen(error_message);
    if (maxlen == 0) return error_message_len;

    size_t write_len = maxlen - 1 < error_message_len ? maxlen - 1 : error_message_len;
    memcpy(message, error_message, write_len * sizeof(char));
    message[write_len] = '\0';
    return write_len;
}