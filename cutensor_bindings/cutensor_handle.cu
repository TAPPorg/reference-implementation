#include "cutensor_bind.h"
#include "../src/tapp/handle.h"

TAPP_EXPORT TAPP_error create_handle(TAPP_handle* handle)//TAPP_error create_TAPP_handle(TAPP_handle* handle)
{
    cutensorHandle_t* cuhandle = new cutensorHandle_t;
    cutensorCreate(cuhandle);
    *handle = (TAPP_handle) cuhandle;
    return 0; // TODO: implement cutensor error handling
}

TAPP_EXPORT TAPP_error TAPP_destroy_handle(TAPP_handle handle)
{
    cutensorHandle_t* cuhandle = (cutensorHandle_t*) handle;
    cutensorDestroy(*cuhandle);
    delete cuhandle;
    return 0; // TODO: implement cutensor error handling
}