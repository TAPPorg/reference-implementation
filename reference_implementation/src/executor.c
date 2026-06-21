/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */
#include "ref_impl.h"
#include <stdlib.h>

TAPP_error TAPP_create_executor(TAPP_executor* exec) {
    *exec = (TAPP_executor)malloc(sizeof(int));
    int ex = 1; // the bruteforce reference executor
#ifdef TAPP_REFERENCE_USE_TBLIS
    // ex = 2; // TBLIS used as executor, use 12 for debug mode
#endif
    *((int*)(*exec)) = ex;
    // exec = (intptr_t)&ex;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec) {
    free((void*)exec);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_executor_get_status(TAPP_executor exec, TAPP_status* status) {
    if (status == NULL) {
        return TAPP_SUCCESS;
    }
    struct status* stat = malloc(sizeof(struct status));
    // All work submitted to a synchronous executor has finished by the time we get here.
    stat->completion = TAPP_COMPLETE;
    stat->error = TAPP_SUCCESS;
    *status = (TAPP_status)stat;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_executor_wait(TAPP_executor exec) {
    // The reference implementation is synchronous; nothing is ever in flight.
    return TAPP_SUCCESS;
}
