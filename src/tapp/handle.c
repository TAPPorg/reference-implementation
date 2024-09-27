/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */
#include "tapp_ex_imp.h"
#include <stdlib.h>

TAPP_error create_handle(TAPP_handle* handle) {
    *handle = (TAPP_handle)malloc(sizeof(TAPP_handle));
    return 0;
}

TAPP_error TAPP_destroy_handle(TAPP_handle handle) {
    free((void*)handle);
    return 0;
}

