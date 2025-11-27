/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */
#include "ref_imp.h"
#include <stdlib.h>

TAPP_error TAPP_create_executor(TAPP_executor* exec) {
    *exec = (TAPP_executor)malloc(sizeof(int));
    int ex = 1; // the bruteforce reference executor
#ifdef ENABLE_TBLIS
    // ex = 2; // TBLIS used as executor, use 12 for debug mode
#endif
    *((int*)(*exec)) = ex;
    // exec = (intptr_t)&ex;
    return 0;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec) {
    free((void*)exec);
    return 0;
}
