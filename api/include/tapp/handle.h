#ifndef TAPP_HANDLE_H_
#define TAPP_HANDLE_H_

#include "util.h"
#include "types.h"


TAPP_EXPORT TAPP_error TAPP_create_handle(uint64_t lib_id, TAPP_handle* handle);

/*
 * TODO: implementation-defined creation of handles or "wrapper" to get all implementations and select one?
 *       devices probably can't be enumerated until you have a handle....
 */

 //TODO: get string describing implementation?

 //TODO: API versioning and query

 //TODO: optional APIs with feature test macros

TAPP_EXPORT TAPP_error TAPP_destroy_handle(TAPP_handle handle);

#endif /* TAPP_HANDLE_H_ */
