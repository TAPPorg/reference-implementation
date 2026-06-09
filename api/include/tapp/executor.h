#ifndef TAPP_EXECUTOR_H_
#define TAPP_EXECUTOR_H_

#include <stdint.h>

#include "util.h"
#include "error.h"
#include "status.h"

typedef intptr_t TAPP_executor;

TAPP_EXPORT TAPP_error TAPP_create_executor(TAPP_executor* exec);

/*
 * TODO: implementation-defined creation of executors or "wrapper" to get all implementations and select one?
 *       devices probably can't be enumerated until you have a handle....
 */

TAPP_EXPORT TAPP_error TAPP_destroy_executor(TAPP_executor exec);

TAPP_EXPORT TAPP_error TAPP_executor_get_status(TAPP_executor exec, TAPP_status* status);

TAPP_EXPORT TAPP_error TAPP_executor_wait(TAPP_executor exec);

#endif /* TAPP_HANDLE_H_ */
