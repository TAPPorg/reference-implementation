#ifndef HI_TAPP_EXECUTOR_H_
#define HI_TAPP_EXECUTOR_H_

#include <stdint.h>

#include "hi_tapp/error.h"

typedef intptr_t HI_TAPP_executor;

/*
 * TODO: implementation-defined creation of executors or "wrapper" to get all implementations and select one?
 *       devices probably can't be enumerated until you have a handle....
 */

HI_TAPP_error HI_TAPP_destroy_executor(HI_TAPP_executor exec);

#endif /* HI_TAPP_HANDLE_H_ */
