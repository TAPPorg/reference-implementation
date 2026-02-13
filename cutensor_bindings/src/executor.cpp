#include "../include/executor.h"

TAPP_error TAPP_create_executor(TAPP_executor* exec)
{
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    cudaError_t cerr;
    cerr = cudaStreamCreate(stream);
    if (cerr != cudaSuccess) return pack_error(0, cerr);
    *exec = (TAPP_executor)stream;
    return pack_error(0, cerr);
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec)
{
    cudaStream_t* stream = (cudaStream_t*)exec;
    cudaError_t cerr;
    cerr = cudaStreamDestroy(*stream);
    if (cerr != cudaSuccess) return pack_error(0, cerr);
    free(stream);
    return pack_error(0, cerr);
}
