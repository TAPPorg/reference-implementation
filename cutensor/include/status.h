#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_STATUS_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_STATUS_H_

#include <tapp/status.h>

#include <cuda_runtime.h>

#include "error.h"

struct status
{
    cudaEvent_t event;
};

// Allocate a status object recording the current point of `stream`.
TAPP_error create_status(cudaStream_t stream, TAPP_status* status);

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_STATUS_H_ */
