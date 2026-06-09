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

TAPP_error TAPP_status_check_completion(TAPP_status status, TAPP_completion* completion)
{
    if (completion == nullptr) return 0;
    cudaError_t cerr = cudaEventQuery(((struct status*)status)->event);
    if (cerr == cudaSuccess)
    {
        *completion = TAPP_COMPLETE;
    }
    else if (cerr == cudaErrorNotReady)
    {
        *completion = TAPP_INCOMPLETE;
    }
    else
    {
        return pack_error(0, cerr);
    }
    return 0;
}

TAPP_error TAPP_status_wait(TAPP_status status)
{
    cudaError_t cerr = cudaEventSynchronize(((struct status*)status)->event);
    if (cerr != cudaSuccess) return pack_error(0, cerr);
    return 0;
}

TAPP_error TAPP_destroy_status(TAPP_status status)
{
    cudaError_t cerr = cudaEventDestroy(((struct status*)status)->event);
    delete (struct status*)status;
    if (cerr != cudaSuccess) return pack_error(0, cerr);
    return 0;
}
