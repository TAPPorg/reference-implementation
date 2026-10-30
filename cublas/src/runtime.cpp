#include "internal.h"

TAPP_error TAPP_start(void) {
    // Nothing to start: CUDA initializes itself on first use
    return 0;
}

TAPP_error TAPP_shutdown(void) {
    return 0;
}
