/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "tapp/runtime.h"

TAPP_error TAPP_start(void) {
    // Nothing to start: CUDA initializes itself on first use
    return 0;
}

TAPP_error TAPP_shutdown(void) {
    return 0;
}
