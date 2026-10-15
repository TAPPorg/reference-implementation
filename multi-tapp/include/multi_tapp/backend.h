#ifndef TAPP_MULTI_TAPP_BACKEND_H_
#define TAPP_MULTI_TAPP_BACKEND_H_

#include <stdint.h>

#include "tapp/util.h"
#include "tapp/error.h"

/* A TAPP implementation loaded by multi-tapp */
typedef intptr_t Multi_TAPP_backend;

/* Load the TAPP implementation in the shared library at libpath */
TAPP_EXPORT TAPP_error Multi_TAPP_create_backend(Multi_TAPP_backend* backend, const char *libpath);

TAPP_EXPORT TAPP_error Multi_TAPP_destroy_backend(Multi_TAPP_backend backend);

/* Start the backend. Should be called before any other use of the backend */
TAPP_EXPORT TAPP_error Multi_TAPP_start(Multi_TAPP_backend backend);

/* Shut down the backend. Should be called after the last use of the backend, before it is destroyed */
TAPP_EXPORT TAPP_error Multi_TAPP_shutdown(Multi_TAPP_backend backend);

#endif /* TAPP_MULTI_TAPP_BACKEND_H_ */
