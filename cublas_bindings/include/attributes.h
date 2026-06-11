#ifndef TAPP_REF_IMPL_CUBLAS_BINDINGS_ATTRIBUTES_H_
#define TAPP_REF_IMPL_CUBLAS_BINDINGS_ATTRIBUTES_H_

#include <tapp/attributes.h>

#include <cstring>

#include "handle.h"

#define ATTR_KEY_USE_DEVICE_MEMORY 0
// Decimal digits of precision requested for cuBLAS fixed-point FP64 emulation
// (int; 0 = emulation off). Read when a tensor product plan is created.
#define ATTR_KEY_PRECISION_DIGITS 1
#define ATTR_COUNT 2

#endif /* TAPP_REF_IMPL_CUBLAS_BINDINGS_ATTRIBUTES_H_ */
