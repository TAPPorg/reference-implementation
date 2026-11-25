#include "internal.h"

TAPP_error TAPP_create_executor(TAPP_executor* exec, TAPP_attr attr)
{
    
    if (!exec) return TAPP_ERR_NULL_POINTER;

    int device = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cutensorHandle_t cutensor = nullptr;
    bool owns_stream = true;

    TAPP_emulation_strategy emulation_strategy = TAPP_DEFAULT_EMULATION_STRATEGY;
    TAPP_emulation_mantissa_control emulation_mantissa_control = TAPP_DEFAULT_EMULATION_MANTISSA_CONTROL;

    if(attr)
    { 
        const attributes& a = *reinterpret_cast<attributes*>(attr);

        bool device_set = false;
        std::map<TAPP_key, std::vector<char>>::const_iterator device_itr = a.find(TAPP_CUBLAS_DEVICE);
        if (device_itr != a.end())
        {
            int n;
            std::memcpy(&n, device_itr->second.data(), sizeof(int));
            device = n;
            device_set = true;
        }

        std::map<TAPP_key, std::vector<char>>::const_iterator stream_itr = a.find(TAPP_CUBLAS_STREAM);
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

        std::map<TAPP_key, std::vector<char>>::const_iterator stategy_itr = a.find(TAPP_CUBLAS_EMULATION_STRATEGY);
        if (stategy_itr != a.end())
        {
            TAPP_emulation_strategy n;
            std::memcpy(&n, stategy_itr->second.data(), sizeof(TAPP_emulation_strategy));
            emulation_strategy = n;
        }

        std::map<TAPP_key, std::vector<char>>::const_iterator mantissa_control_itr = a.find(TAPP_CUBLAS_EMULATION_MANTISSA_CONTROL);
        if (mantissa_control_itr != a.end())
        {
            TAPP_emulation_mantissa_control n;
            std::memcpy(&n, mantissa_control_itr->second.data(), sizeof(TAPP_emulation_mantissa_control));
            emulation_mantissa_control = n;
        }

    }

    device_scope scope(device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    cudaError_t cerr = cudaStreamCreate(&stream);
    if (cerr != cudaSuccess)
    {
        return tapp_error(cerr);
    }

    cublasStatus_t cstat = cublasCreate(&cublas);
    if (cstat != CUBLAS_STATUS_SUCCESS)
    {
        return tapp_error(cstat);
    }

    cutensorStatus_t err = cutensorCreate(&cutensor);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        return tapp_error(err);
    }

    if (owns_stream)
    {
        cerr = cudaStreamCreate(&stream);
        if (cerr != cudaSuccess)
        {
            cutensorDestroy(cutensor);
            cublasDestroy(cublas);
            return tapp_error(cerr);
        }
    }

    struct executor* exec_struct = new struct executor;
    exec_struct->owns_stream = owns_stream;
    exec_struct->stream = stream;
    exec_struct->cublas = cublas;
    exec_struct->device = device;
    exec_struct->cutensor = cutensor;
    exec_struct->emulation_strategy = emulation_strategy;
    exec_struct->emulation_mantissa_control = emulation_mantissa_control;

    *exec = (TAPP_executor)exec_struct;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec)
{
    if (!exec) return TAPP_ERR_NULL_POINTER;

    struct executor* exec_struct = (struct executor*) exec;

    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess)
    {
        // Without its device the GPU resources cannot be released and leak, but the executor is gone either way
        delete exec_struct;
        return tapp_error(scope.error);
    }

    TAPP_error result = TAPP_SUCCESS;
    // Keeps the first error: destroying continues after a failure so nothing else leaks
    auto keep_first = [&result](TAPP_error e) { if (result == TAPP_SUCCESS) result = e; };

    if(exec_struct->owns_stream) keep_first(tapp_error(cudaStreamDestroy(exec_struct->stream)));
    keep_first(tapp_error(cublasDestroy(exec_struct->cublas)));
    keep_first(tapp_error(cutensorDestroy(exec_struct->cutensor)));

    delete exec_struct;

    return result;
}

TAPP_error TAPP_wait_all(TAPP_executor exec)
{
    if (!exec) return TAPP_ERR_NULL_POINTER;

    struct executor* exec_struct = (struct executor*) exec;

    cudaError_t cerr = cudaStreamSynchronize(exec_struct->stream);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_wait_one(TAPP_executor exec, TAPP_status status)
{
    if (!exec || !status) return TAPP_ERR_NULL_POINTER;

    struct status* status_struct = (struct status*) status;

    cudaError_t cerr = cudaEventSynchronize(status_struct->event);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    return TAPP_SUCCESS;
}
