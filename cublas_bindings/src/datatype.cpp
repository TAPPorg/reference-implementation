#include "../include/datatype.h"

DataType translate_datatype(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
        return DataType::FLOAT32;
    case TAPP_F64:
        return DataType::FLOAT64;
    case TAPP_C32:
        return DataType::COMPLEX32;
    case TAPP_C64:
        return DataType::COMPLEX64;
    default:
        // F16/BF16 (and anything else) are unsupported by the TTGT/cuBLAS-GEMM
        // back-end. The product layer rejects these before reaching here; fall
        // back to FLOAT32 to keep this translation total.
        return DataType::FLOAT32;
    }
}

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
