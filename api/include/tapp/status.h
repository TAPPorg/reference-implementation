#ifndef TAPP_STATUS_H_
#define TAPP_STATUS_H_

#include <stdint.h>

#include "util.h"
#include "error.h"

typedef intptr_t TAPP_status;

/*
 * Status objects are created by execution functions (e.g. TAPP_execute_product) and by
 * TAPP_executor_get_status.
 *
 * TODO: how to get data out? using attributes or separate standardized interface? implementation-defined?
 */

typedef int TAPP_completion;

enum
{
    TAPP_INCOMPLETE = 0,
    TAPP_COMPLETE   = 1,
    TAPP_FAILED     = 2,
};

TAPP_EXPORT TAPP_error TAPP_status_check_completion(TAPP_status status,
                                                    TAPP_completion* completion,
                                                    TAPP_error* error);

/* Returns success only if the operation has completed successfully; a failed or
 * not-yet-completed operation returns a non-success error code. */
TAPP_EXPORT TAPP_error TAPP_status_get_error(TAPP_status status);

/* TAPP_status_wait(TAPP_status status) was here; removed because per-operation waiting is
 * implementation-heavy on CPU back-ends. Wait on the executor (TAPP_executor_wait) instead. */

TAPP_EXPORT TAPP_error TAPP_destroy_status(TAPP_status status);

#endif /* TAPP_STATUS_H_ */
