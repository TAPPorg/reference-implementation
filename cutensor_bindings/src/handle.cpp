#include "../include/handle.h"

TAPP_error TAPP_create_handle(TAPP_handle* handle)
{
    cutensorHandle_t* libhandle = new cutensorHandle_t;
    cutensorStatus_t err = cutensorCreate(libhandle);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        delete libhandle;
        return tapp_error(err);
    }
    struct handle* handle_struct = new struct handle;
    handle_struct->libhandle = libhandle;
    bool* use_device_memory = new bool(true);
    handle_struct->attributes = new intptr_t[1];
    handle_struct->attributes[0] = (intptr_t) use_device_memory;
    *handle = (TAPP_handle) handle_struct;
    return TAPP_SUCCESS; 
}

TAPP_error TAPP_destroy_handle(TAPP_handle handle)
{
    struct handle* handle_struct = (struct handle*) handle;
    cutensorStatus_t err = cutensorDestroy(*handle_struct->libhandle);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        return tapp_error(err);
    }
    delete handle_struct->libhandle;
    delete (bool*)handle_struct->attributes[0];
    delete[] handle_struct->attributes;
    delete handle_struct;
    return TAPP_SUCCESS;
}