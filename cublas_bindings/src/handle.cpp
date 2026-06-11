#include "../include/handle.h"

TAPP_error TAPP_create_handle(TAPP_handle* handle)
{
    struct handle* handle_struct = new struct handle;
    bool* use_device_memory = new bool(true);
    handle_struct->attributes = new intptr_t[1];
    handle_struct->attributes[0] = (intptr_t) use_device_memory;
    *handle = (TAPP_handle) handle_struct;
    return 0;
}

TAPP_error TAPP_destroy_handle(TAPP_handle handle)
{
    struct handle* handle_struct = (struct handle*) handle;
    delete (bool*)handle_struct->attributes[0];
    delete[] handle_struct->attributes;
    delete handle_struct;
    return 0;
}
