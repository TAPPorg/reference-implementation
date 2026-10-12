#include "../include/status.h"

TAPP_error create_status(cudaStream_t stream, TAPP_status* status)
{
    if (status == nullptr) return TAPP_SUCCESS;
    struct status* s = new struct status;
    cudaError_t cerr = cudaEventCreate(&s->event);
    if (cerr != cudaSuccess)
    {
        delete s;
        return tapp_error(cerr);
    }
    cerr = cudaEventRecord(s->event, stream);
    if (cerr != cudaSuccess)
    {
        cudaEventDestroy(s->event);
        delete s;
        return tapp_error(cerr);
    }
    *status = (TAPP_status)s;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_status_check_completion(TAPP_status status, TAPP_completion* completion, TAPP_error* error)
{
    cudaError_t cerr = cudaEventQuery(((struct status*)status)->event);
    TAPP_completion comp;
    TAPP_error op_error = TAPP_SUCCESS;
    if (cerr == cudaSuccess)
    {
        comp = TAPP_COMPLETE;
    }
    else if (cerr == cudaErrorNotReady)
    {
        comp = TAPP_INCOMPLETE;
    }
    else
    {
        comp = TAPP_FAILED;
        op_error = tapp_error(cerr);
    }
    if (completion != nullptr) *completion = comp;
    if (error != nullptr) *error = op_error;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_status_get_error(TAPP_status status)
{
    cudaError_t cerr = cudaEventQuery(((struct status*)status)->event);
    if (cerr == cudaSuccess) return TAPP_SUCCESS;
    return tapp_error(cerr);
}

TAPP_error TAPP_destroy_status(TAPP_status status)
{
    cudaError_t cerr = cudaEventDestroy(((struct status*)status)->event);
    delete (struct status*)status;
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}
