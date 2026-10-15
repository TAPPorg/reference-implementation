#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_EXECUTOR_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_EXECUTOR_H_

#include <tapp/executor.h>

#include "error.h"
#include <unordered_map>

struct executor
{
    cudaStream_t* stream;
    cutensorHandle_t* libhandle;
    int* device;
    bool need_alloc; // allocate steam and device if no attributes are provided
};

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_EXECUTOR_H_ */
