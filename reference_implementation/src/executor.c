/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */
#include "internal.h"

TAPP_error TAPP_create_executor(TAPP_executor* exec, TAPP_attr attr) {
    *exec = (TAPP_executor)malloc(sizeof(int));
    int ex = 1; // the bruteforce reference executor
    *((int*)(*exec)) = ex;
    // exec = (intptr_t)&ex;
    return 0;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec) {
    free((void*)exec);
    return 0;
}
