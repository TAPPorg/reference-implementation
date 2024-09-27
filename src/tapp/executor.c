/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */
#include "tapp_ex_imp.h"
#include <stdlib.h>

TAPP_error create_executor(TAPP_executor* exec) {
    *exec = (TAPP_executor)malloc(sizeof(TAPP_executor));
    return 0;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec) {
    free((void*)exec);
    return 0;
}