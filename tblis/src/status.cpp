/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "internal.h"
#include "tapp/status.h"

TAPP_error TAPP_status_get_error(TAPP_status status) {
    if (status == 0) return TAPP_TBLIS_ERROR_NULL_POINTER;
    return *reinterpret_cast<struct status*>(status)->error;
}

TAPP_error TAPP_destroy_status(TAPP_status status) {
    delete reinterpret_cast<struct status*>(status);
    return TAPP_TBLIS_SUCCESS;
}