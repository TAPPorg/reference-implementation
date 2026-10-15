#include "../include/multi_tapp/status.h"
#include "internal.h"

TAPP_error Multi_TAPP_status_get_error(Multi_TAPP_backend backend, TAPP_status status)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (status == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL status\n");
        return MULTI_TAPP_ERROR_INVALID_STATUS;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_status_get_error == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_status_get_error\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_status* multi_tapp_status = (struct mt_status*)status;
    if (backend != multi_tapp_status->backend)
    {
        return MULTI_TAPP_ERROR_STATUS_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_status_get_error(multi_tapp_status->status);

    return multi_tapp_error_from_backend(error);
}

TAPP_error Multi_TAPP_destroy_status(Multi_TAPP_backend backend, TAPP_status status)
{
    if (status == 0)
    {
        fprintf(stderr, "WARNING: Tried to destroy status 0\n");
        return MULTI_TAPP_SUCCESS;
    }
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;

    struct mt_status* multi_tapp_status = (struct mt_status*)status;
    if (backend != multi_tapp_status->backend)
    {
        return MULTI_TAPP_ERROR_STATUS_WRONG_BACKEND;
    }

    // A backend without TAPP_destroy_status has nothing to release for its statuses, only the wrapper is freed
    TAPP_error error = MULTI_TAPP_SUCCESS;
    if (backend_ptr->TAPP_destroy_status != NULL)
    {
        error = backend_ptr->TAPP_destroy_status(multi_tapp_status->status);
    }
    free(multi_tapp_status);

    return multi_tapp_error_from_backend(error);
}
