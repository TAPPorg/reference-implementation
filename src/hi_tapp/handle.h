#ifndef HI_TAPP_HANDLE_H_
#define HI_TAPP_HANDLE_H_

#include <stdint.h>

#include "hi_tapp/error.h"

typedef intptr_t HI_TAPP_handle;

/*
 * TODO: implementation-defined creation of handles or "wrapper" to get all implementations and select one?
 *       devices probably can't be enumerated until you have a handle....
 */

 //TODO: get string describing implementation?

 //TODO: API versioning and query

 //TODO: optional APIs with feature test macros

HI_TAPP_error HI_TAPP_destroy_handle(HI_TAPP_handle handle);

#endif /* HI_TAPP_HANDLE_H_ */
