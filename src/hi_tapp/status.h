#ifndef HI_TAPP_STATUS_H_
#define HI_TAPP_STATUS_H_

#include <stdint.h>

#include "hi_tapp/error.h"

typedef intptr_t HI_TAPP_status;

/*
 * Status objects are created by execution functions (e.g. HI_TAPP_execute_product).
 *
 * TODO: how to get data out? using attributes or separate standardized interface? implementation-defined?
 */

HI_TAPP_error HI_TAPP_destroy_status(HI_TAPP_status status);

#endif /* HI_TAPP_STATUS_H_ */
