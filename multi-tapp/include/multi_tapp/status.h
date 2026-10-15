#ifndef TAPP_MULTI_TAPP_STATUS_H_
#define TAPP_MULTI_TAPP_STATUS_H_

#include "backend.h"
#include "tapp/status.h"

TAPP_EXPORT TAPP_error Multi_TAPP_status_get_error(Multi_TAPP_backend backend, TAPP_status status);

TAPP_EXPORT TAPP_error Multi_TAPP_destroy_status(Multi_TAPP_backend backend, TAPP_status status);

#endif /* TAPP_MULTI_TAPP_STATUS_H_ */
