#include "internal.h"

TAPP_error TAPP_create_executor(TAPP_executor* exec, TAPP_attr attr)
{
    
    if (!exec) return TAPP_ERR_NULL_POINTER;

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    TAPP_emulation_strategy emulation_strategy = TAPP_DEFAULT_EMULATION_STRATEGY;
    TAPP_emulation_mantissa_control emulation_mantissa_control = TAPP_DEFAULT_EMULATION_MANTISSA_CONTROL;
    TAPP_num_gemm num_gemm = TAPP_DEFAULT_NUM_GEMM;

    if(attr)
    { 
        const attributes& a = *reinterpret_cast<attributes*>(attr);

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

        std::map<TAPP_key, std::vector<char>>::const_iterator num_gemm_itr = a.find(TAPP_CUBLAS_EMULATION_NUM_GEMM);
        if (num_gemm_itr != a.end())
        {
            TAPP_num_gemm n;
            std::memcpy(&n, num_gemm_itr->second.data(), sizeof(TAPP_num_gemm));
            num_gemm = n;
        }
    }

    cudaError_t cerr = cudaStreamCreate(&stream);
    if (cerr != cudaSuccess)
    {
        return tapp_error(cerr);
    }

    cublasStatus_t cstat = cublasCreate(&cublas);
    if (cstat != CUBLAS_STATUS_SUCCESS)
    {
        cudaStreamDestroy(stream);
        return tapp_error(cstat);
    }

    cstat = cublasSetStream(cublas, stream);
    if (cstat != CUBLAS_STATUS_SUCCESS)
    {
        cudaStreamDestroy(stream);
        cublasDestroy(cublas);
        return tapp_error(cstat);
    }

    struct executor* exec_struct = new struct executor;
    exec_struct->stream = stream;
    exec_struct->cublas = cublas;
    exec_struct->emulation_strategy = emulation_strategy;
    exec_struct->emulation_mantissa_control = emulation_mantissa_control;
    exec_struct->num_gemm = num_gemm;

    *exec = (TAPP_executor)exec_struct;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec)
{
    if (!exec) return TAPP_ERR_NULL_POINTER;

    struct executor* exec_struct = (struct executor*) exec;

    TAPP_error result = TAPP_SUCCESS;
    // Keeps the first error: destroying continues after a failure so nothing else leaks
    auto keep_first = [&result](TAPP_error e) { if (result == TAPP_SUCCESS) result = e; };

    keep_first(tapp_error(cublasDestroy(exec_struct->cublas)));
    keep_first(tapp_error(cudaStreamDestroy(exec_struct->stream)));

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