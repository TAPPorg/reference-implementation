#include "../include/status.h"

TAPP_error create_status(cudaStream_t stream, TAPP_status* status)
{
    if (status == nullptr) return 0;
    struct status* s = new struct status;
    cudaError_t cerr = cudaEventCreate(&s->event);
    if (cerr != cudaSuccess)
    {
        delete s;
        return pack_error(0, cerr);
    }
    cerr = cudaEventRecord(s->event, stream);
    if (cerr != cudaSuccess)
    {
        cudaEventDestroy(s->event);
        delete s;
        return pack_error(0, cerr);
    }
    *status = (TAPP_status)s;
    return 0;
}

TAPP_error TAPP_status_check_completion(TAPP_status status, TAPP_completion* completion, TAPP_error* error)
{
    cudaError_t cerr = cudaEventQuery(((struct status*)status)->event);
    TAPP_completion comp;
    TAPP_error op_error = 0;
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
        op_error = pack_error(0, cerr);
    }
    if (completion != nullptr) *completion = comp;
    if (error != nullptr) *error = op_error;
    return 0;
}

TAPP_error TAPP_status_get_error(TAPP_status status)
{
    cudaError_t cerr = cudaEventQuery(((struct status*)status)->event);
    if (cerr == cudaSuccess) return 0;
    return pack_error(0, cerr);
}

TAPP_error TAPP_destroy_status(TAPP_status status)
{
    cudaError_t cerr = cudaEventDestroy(((struct status*)status)->event);
    delete (struct status*)status;
    if (cerr != cudaSuccess) return pack_error(0, cerr);
    return 0;
}
