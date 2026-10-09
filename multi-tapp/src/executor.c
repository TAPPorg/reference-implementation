#include "../include/executor.h"

TAPP_error Multi_TAPP_create_executor(TAPP_backend backend,
                                      TAPP_executor* exec,
                                      TAPP_attr attr)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_create_executor == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_executor\n");
        return 6; // TODO: Return error for non implemented function
    }
    
    struct Multi_TAPP_attr* multi_tapp_attr = (struct Multi_TAPP_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return 13; // TODO: Return error for incompatible attribute
    }

    struct Multi_TAPP_executor* multi_tapp_exec = malloc(sizeof(struct Multi_TAPP_executor));
    multi_tapp_exec->backend = backend;
    *exec = (TAPP_executor)multi_tapp_exec;

    return backend_ptr->TAPP_create_executor(&multi_tapp_exec->exec, multi_tapp_attr->attr);
}

TAPP_error Multi_TAPP_destroy_executor(TAPP_backend backend,
                                       TAPP_executor exec)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_destroy_executor == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_executor\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_executor* multi_tapp_exec = (struct Multi_TAPP_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    TAPP_error error = backend_ptr->TAPP_destroy_executor(multi_tapp_exec->exec);
    free(multi_tapp_exec);

    return error;
}

TAPP_error Multi_TAPP_wait_one(TAPP_backend backend, TAPP_executor exec, TAPP_status status)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_wait_one == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_get\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_executor* multi_tapp_executor = (struct Multi_TAPP_executor*)exec;
    if (backend != multi_tapp_executor->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    TAPP_error error = backend_ptr->TAPP_wait_one(exec, status);

    return error;
}

TAPP_error Multi_TAPP_wait_all(TAPP_backend backend, TAPP_executor exec)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_wait_all == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_get\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_executor* multi_tapp_executor = (struct Multi_TAPP_executor*)exec;
    if (backend != multi_tapp_executor->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    TAPP_error error = backend_ptr->TAPP_wait_all(exec);

    return error;
}