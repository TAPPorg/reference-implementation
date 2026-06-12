#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_ERROR_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_ERROR_H_

#include <tapp/error.h>

#include <cutensor.h>

#include <cstring>
#include <string>

inline TAPP_error tapp_error(cudaError_t e)      { return e == cudaSuccess ? TAPP_SUCCESS : tapp_error(TAPP_ERROR_TYPE_CUDA, (int)e); }
inline TAPP_error tapp_error(cutensorStatus_t e) { return e == CUTENSOR_STATUS_SUCCESS ? TAPP_SUCCESS : tapp_error(TAPP_ERROR_TYPE_CUTENSOR, (int)e); }

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDS_ERROR_H_ */