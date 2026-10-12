#ifndef TAPP_MULTI_TAPP_TENSOR_H_
#define TAPP_MULTI_TAPP_TENSOR_H_

#include "backend.h"
#include "tapp/tensor.h"

TAPP_EXPORT TAPP_error Multi_TAPP_create_tensor_info(TAPP_backend backend,
                                                     TAPP_tensor_info* info,
                                                     TAPP_datatype type,
                                                     int nmode,
                                                     const int64_t* extents,
                                                     const int64_t* strides);

TAPP_EXPORT TAPP_error Multi_TAPP_destroy_tensor_info(TAPP_backend backend, TAPP_tensor_info info);

TAPP_EXPORT int Multi_TAPP_get_nmodes(TAPP_backend backend, TAPP_tensor_info info);

TAPP_EXPORT TAPP_error Multi_TAPP_set_nmodes(TAPP_backend backend,
                                             TAPP_tensor_info info,
                                             int nmodes);

TAPP_EXPORT void Multi_TAPP_get_extents(TAPP_backend backend,
                                        TAPP_tensor_info info,
                                        int64_t* extents);

TAPP_EXPORT TAPP_error Multi_TAPP_set_extents(TAPP_backend backend,
                                              TAPP_tensor_info info,
                                              const int64_t* extents);

TAPP_EXPORT void Multi_TAPP_get_strides(TAPP_backend backend,
                                        TAPP_tensor_info info,
                                        int64_t* strides);

TAPP_EXPORT TAPP_error Multi_TAPP_set_strides(TAPP_backend backend,
                                              TAPP_tensor_info info,
                                              const int64_t* strides);

#endif /* TAPP_MULTI_TAPP_TENSOR_H_ */