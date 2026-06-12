#include <tapp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    TAPP_handle h;
    TAPP_error err = TAPP_create_handle(&h);
    if (!TAPP_check_success(err)) {
        fprintf(stderr, "TAPP_create_handle failed: type %d code %d\n", err.type, err.code);
        return EXIT_FAILURE;
    }
    err = TAPP_destroy_handle(h);
    if (!TAPP_check_success(err)) {
        fprintf(stderr, "TAPP_destroy_handle failed: type %d code %d\n", err.type, err.code);
        return EXIT_FAILURE;
    }
    printf("tapp-reference smoke test passed\n");
    return EXIT_SUCCESS;
}
