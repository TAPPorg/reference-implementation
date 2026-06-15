#include "../include/handle.h"
#include "../include/attributes.h"

TAPP_error TAPP_create_handle(TAPP_handle* handle)
{
    struct handle* handle_struct = new struct handle;
    cublasStatus_t stat = cublasCreate(&handle_struct->cublas);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        delete handle_struct;
        return tapp_error(stat);
    }
    handle_struct->attributes = new intptr_t[ATTR_COUNT];
    handle_struct->attributes[ATTR_KEY_USE_DEVICE_MEMORY] = (intptr_t) new bool(true);
    *handle = (TAPP_handle) handle_struct;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_handle(TAPP_handle handle)
{
    struct handle* handle_struct = (struct handle*) handle;
    cublasStatus_t stat = cublasDestroy(handle_struct->cublas);
    delete (bool*)handle_struct->attributes[ATTR_KEY_USE_DEVICE_MEMORY];
    delete[] handle_struct->attributes;
    delete handle_struct;
    if (stat != CUBLAS_STATUS_SUCCESS) return tapp_error(stat);
    return TAPP_SUCCESS;
}
