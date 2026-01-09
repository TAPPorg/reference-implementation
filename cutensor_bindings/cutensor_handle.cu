#include "cutensor_bind.h"
#include "../src/tapp/handle.h"

TAPP_EXPORT TAPP_error create_handle(TAPP_handle* handle)//TAPP_error create_TAPP_handle(TAPP_handle* handle)
{
    cutensorHandle_t* libhandle = new cutensorHandle_t;
    cutensorCreate(libhandle);
    struct handle* handle_struct = new struct handle;
    handle_struct->libhandle = libhandle;
    bool* use_device_memory = new bool(true);
    handle_struct->attributes = new intptr_t[1];
    handle_struct->attributes[0] = (intptr_t) use_device_memory;
    *handle = (TAPP_handle) handle_struct;
    return 0; // TODO: implement cutensor error handling
}

TAPP_EXPORT TAPP_error TAPP_destroy_handle(TAPP_handle handle)
{
    struct handle* handle_struct = (struct handle*) handle;
    cutensorDestroy(*handle_struct->libhandle);
    delete handle_struct->libhandle;
    delete (bool*)handle_struct->attributes[0];
    delete[] handle_struct->attributes;
    delete handle_struct;
    return 0; // TODO: implement cutensor error handling
}