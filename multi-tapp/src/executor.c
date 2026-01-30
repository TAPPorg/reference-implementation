#include "../include/executor.h"

TAPP_error TAPP_create_executor(TAPP_executor* exec,
                                TAPP_handle handle)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_create_executor == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_executor\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_executor* multi_tapp_exec = malloc(sizeof(struct Multi_TAPP_executor));
    multi_tapp_exec->impl_id = multi_tapp_handle->impl_id;
    *exec = (TAPP_executor)multi_tapp_exec;

    return multi_tapp_handle->TAPP_create_executor(&multi_tapp_exec->exec, multi_tapp_handle->tapp_handle);
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec,
                                TAPP_handle handle)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_destroy_executor == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_executor\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_executor* multi_tapp_exec = (struct Multi_TAPP_executor*)exec;
    if (multi_tapp_handle->impl_id != multi_tapp_exec->impl_id)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    TAPP_error error = multi_tapp_handle->TAPP_destroy_executor(multi_tapp_exec->exec, multi_tapp_handle->tapp_handle);
    free(multi_tapp_exec);

    return error;
}