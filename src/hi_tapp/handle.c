/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */
#include "hi_tapp/tapp_ex_imp.h"
#include <stdlib.h>

HI_TAPP_error create_handle(HI_TAPP_handle* handle) {
    *handle = (HI_TAPP_handle)malloc(sizeof(HI_TAPP_handle));
    return 0;
}

HI_TAPP_error HI_TAPP_destroy_handle(HI_TAPP_handle handle) {
    free((void*)handle);
    return 0;
}

