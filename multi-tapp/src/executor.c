#include "../include/multi_tapp/executor.h"
#include "internal.h"

TAPP_error Multi_TAPP_create_executor(TAPP_backend backend,
                                      TAPP_executor* exec,
                                      TAPP_attr attr)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }

    if (exec == NULL)
    {
        fprintf(stderr, "ERROR: Called function with NULL exec pointer");
        return MULTI_TAPP_ERROR_INVALID_ARGUMENT;
    }

    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_create_executor == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_executor\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }
    
    struct Multi_TAPP_attr* multi_tapp_attr = NULL;
    if (attr != 0)
    {
        multi_tapp_attr = (struct Multi_TAPP_attr*)attr;
        if (backend != multi_tapp_attr->backend)
        {
            return MULTI_TAPP_ERROR_ATTR_WRONG_BACKEND;
        }
    }

    struct Multi_TAPP_executor* multi_tapp_exec = malloc(sizeof(struct Multi_TAPP_executor));
    if (multi_tapp_exec == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate space for an executor object\n");
        return MULTI_TAPP_ERROR_ALLOCATION_FAILED;
    }
    multi_tapp_exec->backend = backend;

    TAPP_error error = backend_ptr->TAPP_create_executor(&multi_tapp_exec->exec, attr == 0 ? 0 : multi_tapp_attr->attr);

    if (!multi_tapp_check_success(error)) {
        free(multi_tapp_exec);
        return multi_tapp_error_from_backend(error);
    }

    *exec = (TAPP_executor)multi_tapp_exec;
    return  MULTI_TAPP_SUCCESS;
}

TAPP_error Multi_TAPP_destroy_executor(TAPP_backend backend,
                                       TAPP_executor exec)
{
    if (exec == 0)
    {
        fprintf(stderr, "WARNING: Tried to destroy exec 0\n");
        return MULTI_TAPP_SUCCESS;
    }
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_destroy_executor == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_executor\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct Multi_TAPP_executor* multi_tapp_exec = (struct Multi_TAPP_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_destroy_executor(multi_tapp_exec->exec);
    free(multi_tapp_exec);

    return multi_tapp_error_from_backend(error);
}

TAPP_error Multi_TAPP_wait_one(TAPP_backend backend, TAPP_executor exec, TAPP_status status)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (exec == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL executor\n");
        return MULTI_TAPP_ERROR_INVALID_EXECUTOR;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_wait_one == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_wait_one\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct Multi_TAPP_executor* multi_tapp_exec = (struct Multi_TAPP_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_wait_one(multi_tapp_exec->exec, status);

    return multi_tapp_error_from_backend(error);
}

TAPP_error Multi_TAPP_wait_all(TAPP_backend backend, TAPP_executor exec)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (exec == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL executor\n");
        return MULTI_TAPP_ERROR_INVALID_EXECUTOR;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_wait_all == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_wait_all\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct Multi_TAPP_executor* multi_tapp_exec = (struct Multi_TAPP_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_wait_all(multi_tapp_exec->exec);

    return multi_tapp_error_from_backend(error);
}