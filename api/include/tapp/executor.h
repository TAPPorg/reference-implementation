#ifndef TAPP_EXECUTOR_H_
#define TAPP_EXECUTOR_H_

#include "util.h"
#include "types.h"

TAPP_EXPORT TAPP_error TAPP_create_executor(TAPP_executor* exec);

/*
 * TODO: implementation-defined creation of executors or "wrapper" to get all implementations and select one?
 *       devices probably can't be enumerated until you have a handle....
 */

TAPP_EXPORT TAPP_error TAPP_destroy_executor(TAPP_executor exec);

#endif /* TAPP_HANDLE_H_ */
