#include "cutensor_bind.h"

TAPP_EXPORT TAPP_error create_executor(TAPP_executor* exec) {
    *exec = (TAPP_executor)malloc(sizeof(int));
    int ex = 1; // the bruteforce reference executor
    *((int*)(*exec)) = ex;
    // exec = (intptr_t)&ex;
    return 0;
}

TAPP_EXPORT TAPP_error TAPP_destroy_executor(TAPP_executor exec) {
    free((void*)exec);
    return 0;
}
