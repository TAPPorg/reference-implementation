/*
 * Ed Valeev
 */
#include "ref_impl.h"
#include <stdlib.h>

TAPP_error TAPP_status_check_completion(TAPP_status status, TAPP_completion* completion, TAPP_error* error) {
    if (completion != NULL) {
        *completion = ((struct status*)status)->completion;
    }
    if (error != NULL) {
        *error = ((struct status*)status)->error;
    }
    return 0;
}

TAPP_error TAPP_status_get_error(TAPP_status status) {
    return ((struct status*)status)->error;
}

TAPP_error TAPP_status_wait(TAPP_status status) {
    // The reference implementation is synchronous, so the work a status refers to has
    // already completed by the time the status exists; there is nothing to wait for.
    return 0;
}

TAPP_error TAPP_destroy_status(TAPP_status status) {
    free((struct status*)status);
    return 0;
}

