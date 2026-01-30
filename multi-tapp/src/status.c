#include "../include/status.h"

TAPP_error TAPP_destroy_status(TAPP_status status,
                               TAPP_handle handle)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_destroy_status == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_status\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_status* multi_tapp_status = (struct Multi_TAPP_status*)status;
    if (multi_tapp_handle->impl_id != multi_tapp_status->impl_id)
    {
        return 14; // TODO: Return error for incompatible handle
    }

    TAPP_error error = multi_tapp_handle->TAPP_destroy_status(*multi_tapp_status->status, *multi_tapp_handle->tapp_handle);
    free(multi_tapp_status);

    return error;
}