#ifndef TAPP_MULTI_TAPP_BACKEND_H_
#define TAPP_MULTI_TAPP_BACKEND_H_

#include <stdint.h>

#include "tapp/util.h"
#include "tapp/error.h"

/* A TAPP implementation loaded by multi-tapp */
typedef intptr_t TAPP_backend;

/* Load the TAPP implementation in the shared library at libpath */
TAPP_EXPORT TAPP_error Multi_TAPP_create_backend(TAPP_backend* backend, const char *libpath);

TAPP_EXPORT TAPP_error Multi_TAPP_destroy_backend(TAPP_backend backend);

#endif /* TAPP_MULTI_TAPP_BACKEND_H_ */
