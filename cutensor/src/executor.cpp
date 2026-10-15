#include "../include/executor.h"
#include "../include/status.h"
#include "../include/attributes.h"

TAPP_error TAPP_create_executor(TAPP_executor* exec, TAPP_attr attr)
{
    struct executor* exec_struct = new struct executor;  

    cutensorHandle_t* libhandle = new cutensorHandle_t;
    cutensorStatus_t err = cutensorCreate(libhandle);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        delete libhandle;
        return tapp_error(err);
    }
    exec_struct->libhandle = libhandle;

    if(attr){ 
        size_t size;
        exec_struct->device = new int;
        TAPP_error terr = TAPP_attr_get(attr, TAPP_CUTENSOR_DEVICE, &size, (void**) &(exec_struct->device));
        if(terr!=TAPP_SUCCESS) return terr;
        terr = TAPP_attr_get(attr, TAPP_CUTENSOR_STREAM, &size, (void**) &(exec_struct->stream));
        if(terr!=TAPP_SUCCESS) return terr;

        exec_struct->need_alloc = false; 
    }else{
        int* device_id = new int;
        *device_id = 0;
        exec_struct->device = device_id;
        cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
        cudaError_t cerr = cudaStreamCreate(stream);
        if (cerr != cudaSuccess) return tapp_error(cerr);
        exec_struct->stream = stream;

        exec_struct->need_alloc = true; 
    }

    *exec = (TAPP_executor)exec_struct;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec)
{
    struct executor* exec_struct = (struct executor*) exec;

    cudaError_t cerr = cudaSetDevice(*(exec_struct->device));
    if (cerr != cudaSuccess) return tapp_error(cerr);

    if(exec_struct->need_alloc){
        delete exec_struct->device;

        cudaError_t cerr = cudaStreamDestroy(*(exec_struct->stream));
        if (cerr != cudaSuccess) return tapp_error(cerr);
        free(exec_struct->stream);
    }

    cutensorStatus_t err = cutensorDestroy(*(exec_struct->libhandle));
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    delete exec_struct->libhandle;

    delete exec_struct;

    return TAPP_SUCCESS;
}

TAPP_error TAPP_executor_get_status(TAPP_executor exec, TAPP_status* status)
{
    struct executor* exec_struct = (struct executor*) exec;
    cudaError_t cerr = cudaSetDevice(*(exec_struct->device));
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return create_status(*(exec_struct->stream), status);
}

TAPP_error TAPP_wait_all(TAPP_executor exec)
{
    struct executor* exec_struct = (struct executor*) exec;
    cudaError_t cerr = cudaSetDevice(*(exec_struct->device));
    if (cerr != cudaSuccess) return tapp_error(cerr);

    cerr = cudaStreamSynchronize(*((exec_struct->stream)));
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_wait_one(TAPP_executor exec, TAPP_status status)
{
    struct executor* exec_struct = (struct executor*) exec;
    cudaError_t cerr = cudaSetDevice(*(exec_struct->device));
    if (cerr != cudaSuccess) return tapp_error(cerr);

    cerr = cudaEventSynchronize(((struct status*) status)->event);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}
