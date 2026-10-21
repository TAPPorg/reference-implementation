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

    cudaError_t cerr = cudaSetDevice(device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    cutensorHandle_t libhandle;
    cutensorStatus_t err = cutensorCreate(&libhandle);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        return tapp_error(err);
    }

    if (owns_stream)
    {
        cerr = cudaStreamCreate(&stream);
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
    struct executor* exec_struct = (struct executor*) exec;

    cudaError_t cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

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
    struct executor* exec_struct = (struct executor*) exec;
    cudaError_t cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return create_status(exec_struct->stream, exec_struct->device, status);
}

TAPP_error TAPP_wait_all(TAPP_executor exec)
{
    struct executor* exec_struct = (struct executor*) exec;
    cudaError_t cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    cerr = cudaStreamSynchronize(exec_struct->stream);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_wait_one(TAPP_executor exec, TAPP_status status)
{
    struct executor* exec_struct = (struct executor*) exec;
    cudaError_t cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    cerr = cudaEventSynchronize(((struct status*) status)->event);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}
