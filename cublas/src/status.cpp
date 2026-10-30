#include "internal.h"

TAPP_error create_status(cudaStream_t stream, TAPP_status* status)
{
    if (status == nullptr) return TAPP_SUCCESS;
    struct status* s = new (std::nothrow) struct status;
    if (!s) return TAPP_ERR_OUT_OF_MEMORY;

    // A status only needs to know when the operation is done, and events without timing are cheaper
    cudaError_t cerr = cudaEventCreateWithFlags(&s->event, cudaEventDisableTiming);
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

TAPP_error TAPP_status_get_error(TAPP_status status)
{
    if (!status) return TAPP_ERR_NULL_POINTER;

    // A failed kernel leaves a sticky error that CUDA reports for every later call, so after such a failure this
    // reports it for every status, not only for the one of the operation that failed
    cudaError_t cerr = cudaEventQuery(((struct status*)status)->event);
    if (cerr == cudaSuccess) return TAPP_SUCCESS;
    if (cerr == cudaErrorNotReady) return TAPP_ERR_STATUS_RUNNING;
    return tapp_error(cerr);
}

TAPP_error TAPP_destroy_status(TAPP_status status)
{
    if (!status) return TAPP_ERR_NULL_POINTER;

    cudaError_t cerr = cudaEventDestroy(((struct status*)status)->event);
    delete (struct status*)status;
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}
