#include "cutensor_bind.h"

TAPP_EXPORT TAPP_error create_executor(TAPP_executor* exec)
{
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    HANDLE_CUDA_ERROR(cudaStreamCreate(stream));
    *exec = (TAPP_executor)stream;
    return 0;
}

TAPP_EXPORT TAPP_error TAPP_destroy_executor(TAPP_executor exec)
{
    cudaStream_t* stream = (cudaStream_t*)exec;
    HANDLE_CUDA_ERROR(cudaStreamDestroy(*stream));
    free(stream);
    return 0;
}
