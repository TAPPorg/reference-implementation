#include "../include/executor.h"
#include "../include/status.h"

TAPP_error TAPP_create_executor(TAPP_executor* exec)
{

    struct executor* exec_struct = new struct executor;  

    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    cudaError_t cerr = cudaStreamCreate(stream);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    exec_struct->stream = stream;

    cutensorHandle_t* libhandle = new cutensorHandle_t;
    cutensorStatus_t err = cutensorCreate(libhandle);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        delete libhandle;
        return tapp_error(err);
    }
    exec_struct->libhandle = libhandle;

    bool* use_device_memory = new bool(true);
    exec_struct->attributes = new intptr_t[1];
    exec_struct->attributes[0] = (intptr_t) use_device_memory;

    *exec = (TAPP_executor)exec_struct;
    return tapp_error(cerr);
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec)
{
    struct executor* exec_struct = (struct executor*) exec;

    cudaError_t cerr = cudaStreamDestroy(*(exec_struct->stream));
    if (cerr != cudaSuccess) return tapp_error(cerr);
    free(exec_struct->stream);

    cutensorStatus_t err = cutensorDestroy(*(exec_struct->libhandle));
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    delete exec_struct->libhandle;

    delete (bool*)exec_struct->attributes[0];
    delete[] exec_struct->attributes;
    delete exec_struct;

    return tapp_error(cerr);
}

TAPP_error TAPP_executor_get_status(TAPP_executor exec, TAPP_status* status)
{
    return create_status(*(((struct executor*) exec)->stream), status);
}

TAPP_error TAPP_wait_executor(TAPP_executor exec)
{
    cudaError_t cerr = cudaStreamSynchronize(*(((struct executor*)exec)->stream));
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}
