#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_EXECUTOR_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_EXECUTOR_H_

#include <tapp/executor.h>

#include "error.h"
#include <unordered_map>

struct executor
{
    cudaStream_t* stream;
    cutensorHandle_t* libhandle;
    // intptr_t* attributes;
};

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_EXECUTOR_H_ */
