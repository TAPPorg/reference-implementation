#include "internal.h"

int tapp_prec_digits(TAPP_prectype prec)
{
    if ((prec >= TAPP_F_3_DIGITS && prec <= TAPP_F_34_DIGITS) ||
        (prec >= TAPP_C_3_DIGITS && prec <= TAPP_C_34_DIGITS))
        return prec % 1000;
    return 0;
}

bool translate_prectype(TAPP_prectype prec, TAPP_datatype datatype, cublasComputeType_t& out)
{
    // Variable-precision (digit-count) types map to cuBLAS fixed-point FP64
    // emulation. Without an emulation-capable build (CUDA >= 13) fall back to
    // plain FP64, i.e. full precision ignoring the requested digit reduction.
    if (tapp_prec_digits(prec) > 0)
    {
#if OZAKI_EMULATION_VERSION==1
        out = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
#else
        out = CUBLAS_COMPUTE_64F;
#endif
        return true;
    }

    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (datatype)
        {
        case TAPP_F32:
        case TAPP_C32:
            out = CUBLAS_COMPUTE_32F;
            return true;
        case TAPP_F64:
        case TAPP_C64:
            out = CUBLAS_COMPUTE_64F;
            return true;
        default:
            return false;
        }
    case TAPP_F32F32_ACCUM_F32:
        out = CUBLAS_COMPUTE_32F;
        return true;
    case TAPP_F64F64_ACCUM_F64:
        out = CUBLAS_COMPUTE_64F;
        return true;
    }
    return false;
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

cutensorDataType_t get_cutensor_datatype(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
        return CUTENSOR_R_32F;
    case TAPP_F64:
        return CUTENSOR_R_64F;
    case TAPP_C32:
        return CUTENSOR_C_32F;
    case TAPP_C64:
        return CUTENSOR_C_64F;
    default:
        return CUTENSOR_R_32F;
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

bool translate_prectype_permute(TAPP_datatype datatype, cutensorComputeDescriptor_t& out)
{
    switch (datatype)
    {
    case TAPP_F32:
    case TAPP_C32:
        out = CUTENSOR_COMPUTE_DESC_32F;
        return true; 
        break;
    case TAPP_F64:
    case TAPP_C64:
        out = CUTENSOR_COMPUTE_DESC_64F;
        return true;
        break;
    }
    return false;
}

bool is_complex(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
        return false;
    case TAPP_F64:
        return false;
    case TAPP_C32:
        return true;
    case TAPP_C64:
        return true;
    default:
        return false;
    }
}

bool translate_prectype(TAPP_prectype in, cutensorDataType_t datatype, cutensorComputeDescriptor_t& out)
{
    switch (in)
    {
        case TAPP_DEFAULT_PREC:
            switch (datatype)
            {
                case CUTENSOR_R_32F:
                case CUTENSOR_C_32F:  out = CUTENSOR_COMPUTE_DESC_32F; return true;
                case CUTENSOR_R_64F:
                case CUTENSOR_C_64F:  out = CUTENSOR_COMPUTE_DESC_64F; return true;
                default: return false; // F16 and BF16 are not supported yet, see translate_datatype
            }
        case TAPP_F32F32_ACCUM_F32: out = CUTENSOR_COMPUTE_DESC_32F; return true;
        case TAPP_F64F64_ACCUM_F64: out = CUTENSOR_COMPUTE_DESC_64F; return true;
        default: return false; // TAPP_F16F16_ACCUM_F16 waits for F16 support, see translate_datatype
    }
}
