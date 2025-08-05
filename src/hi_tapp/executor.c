/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */
#include "hi_tapp/tapp_ex_imp.h"
#include <stdlib.h>

HI_TAPP_error create_executor(HI_TAPP_executor* exec) {
    *exec = (HI_TAPP_executor)malloc(sizeof(int));
    int ex = 1;
    *((int*)(*exec)) = ex;
    // exec = (intptr_t)&ex;
    return 0;
}

HI_TAPP_error HI_TAPP_destroy_executor(HI_TAPP_executor exec) {
    free((void*)exec);
    return 0;
}
