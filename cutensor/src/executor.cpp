#include "internal.h"

TAPP_error TAPP_create_executor(TAPP_executor* exec, TAPP_attr attr)
{
    if (!exec) return TAPP_ERR_NULL_POINTER;

    int device = 0;
    cudaStream_t stream = nullptr;
    bool owns_stream = true;

    if(attr)
    { 
        const attributes& a = *reinterpret_cast<attributes*>(attr);

        bool device_set = false;
        std::map<TAPP_key, std::vector<char>>::const_iterator device_itr = a.find(TAPP_CUTENSOR_DEVICE);
        if (device_itr != a.end())
        {
            int n;
            std::memcpy(&n, device_itr->second.data(), sizeof(int));
            device = n;
            device_set = true;
        }

        std::map<TAPP_key, std::vector<char>>::const_iterator stream_itr = a.find(TAPP_CUTENSOR_STREAM);
        if (stream_itr != a.end())
        {
            cudaStream_t n;
            std::memcpy(&n, stream_itr->second.data(), sizeof(cudaStream_t));
            stream = n;
            owns_stream = false;
#if CUDART_VERSION >= 12080
            int stream_device;
            cudaError_t cerr = cudaStreamGetDevice(stream, &stream_device);
            if (cerr != cudaSuccess) return tapp_error(cerr);
            if (device_set && device != stream_device)
            {
                return TAPP_ERR_ATTR_INVALID_STREAM;
            }
            else
            {
                device = stream_device;
            }
#endif
        }
    }

    // The cuTENSOR handle and the stream belong to the device that is current when they are created
    device_scope scope(device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    cutensorHandle_t libhandle;
    cutensorStatus_t err = cutensorCreate(&libhandle);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        return tapp_error(err);
    }

    if (owns_stream)
    {
        cudaError_t cerr = cudaStreamCreate(&stream);
        if (cerr != cudaSuccess)
        {
            cutensorDestroy(libhandle);
            return tapp_error(cerr);
        }
    }

    struct executor* exec_struct = new struct executor;
    exec_struct->libhandle = libhandle;
    exec_struct->stream = stream;
    exec_struct->owns_stream = owns_stream;
    exec_struct->device = device;

    *exec = (TAPP_executor)exec_struct;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec)
{
    if (!exec) return TAPP_ERR_NULL_POINTER;

    struct executor* exec_struct = (struct executor*) exec;

    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    if (exec_struct->workspace)
    {
        cudaFreeAsync(exec_struct->workspace, exec_struct->stream);
        // A stream we do not own is not destroyed below, so wait here until the free has happened
        if (!exec_struct->owns_stream) cudaStreamSynchronize(exec_struct->stream);
    }

    if(exec_struct->owns_stream){
        cudaError_t cerr = cudaStreamDestroy(exec_struct->stream);
        if (cerr != cudaSuccess) return tapp_error(cerr);
    }

    cutensorStatus_t err = cutensorDestroy(exec_struct->libhandle);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    delete exec_struct;

    return TAPP_SUCCESS;
}

TAPP_error TAPP_executor_get_status(TAPP_executor exec, TAPP_status* status)
{
    if (!exec) return TAPP_ERR_NULL_POINTER;

    struct executor* exec_struct = (struct executor*) exec;
    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);
    return create_status(exec_struct->stream, exec_struct->device, status);
}

TAPP_error TAPP_wait_all(TAPP_executor exec)
{
    if (!exec) return TAPP_ERR_NULL_POINTER;

    struct executor* exec_struct = (struct executor*) exec;
    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    cudaError_t cerr = cudaStreamSynchronize(exec_struct->stream);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_wait_one(TAPP_executor exec, TAPP_status status)
{
    if (!exec || !status) return TAPP_ERR_NULL_POINTER;

    struct executor* exec_struct = (struct executor*) exec;
    struct status* status_struct = (struct status*) status;
    if (status_struct->device != exec_struct->device) return TAPP_ERR_STATUS_WRONG_DEVICE;

    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    cudaError_t cerr = cudaEventSynchronize(status_struct->event);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}
