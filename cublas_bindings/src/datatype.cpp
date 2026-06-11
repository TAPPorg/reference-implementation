#include "../include/datatype.h"

cublasComputeType_t translate_prectype(TAPP_prectype prec, TAPP_datatype datatype)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (datatype)
        {
        case TAPP_F32:
        case TAPP_C32:
            return CUBLAS_COMPUTE_32F;
        case TAPP_F64:
        case TAPP_C64:
            return CUBLAS_COMPUTE_64F;
        default:
            return CUBLAS_COMPUTE_32F;
        }
    case TAPP_F32F32_ACCUM_F32:
        return CUBLAS_COMPUTE_32F;
    case TAPP_F64F64_ACCUM_F64:
        return CUBLAS_COMPUTE_64F;
    default:
        return CUBLAS_COMPUTE_32F;
    }
}

cudaDataType get_cuda_datatype(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
        return CUDA_R_32F;
    case TAPP_F64:
        return CUDA_R_64F;
    case TAPP_C32:
        return CUDA_C_32F;
    case TAPP_C64:
        return CUDA_C_64F;
    default:
        return CUDA_R_32F;
    }
}

size_t sizeof_datatype(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
        return sizeof(float);
    case TAPP_F64:
        return sizeof(double);
    case TAPP_C32:
        return sizeof(std::complex<float>);
    case TAPP_C64:
        return sizeof(std::complex<double>);
    default:
        return sizeof(float);
    }
}
