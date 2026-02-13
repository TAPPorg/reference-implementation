#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_HANDLE_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_HANDLE_H_

#include <tapp/handle.h>

#include <cutensor.h>

#include "error.h"

struct handle
{
    cutensorHandle_t* libhandle;
    intptr_t* attributes;
};

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_HANDLE_H_ */