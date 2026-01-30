#ifndef TAPP_STATUS_H_
#define TAPP_STATUS_H_

#include "util.h"
#include "types.h"

/*
 * Status objects are created by execution functions (e.g. TAPP_execute_product).
 *
 * TODO: how to get data out? using attributes or separate standardized interface? implementation-defined?
 */

TAPP_EXPORT TAPP_error TAPP_destroy_status(TAPP_status status, TAPP_handle handle);

#endif /* TAPP_STATUS_H_ */
