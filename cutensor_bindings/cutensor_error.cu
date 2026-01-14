#include "cutensor_bind.h"

// pack multiple types of error codes into one int
constexpr int TAPP_BITS   = 5;
constexpr int CUTENSOR_BITS = 9;
constexpr int CUTENSOR_OFFS = TAPP_BITS;    // 5
constexpr int CUDA_OFFS   = CUTENSOR_OFFS + CUTENSOR_BITS; // 14
constexpr uint64_t TAPP_FIELD_MASK   = (1ULL << TAPP_BITS) - 1; // 0x1F
constexpr uint64_t CUTENSOR_FIELD_MASK = ((1ULL << CUTENSOR_BITS) - 1) << CUTENSOR_OFFS;
constexpr uint64_t TAPP_CLEAR_MASK   = ~TAPP_FIELD_MASK;
constexpr uint64_t CUTENSOR_CLEAR_MASK = ~CUTENSOR_FIELD_MASK;


bool TAPP_check_success(TAPP_error error) {
    return error == 0;
}


size_t TAPP_explain_error(TAPP_error error,
                          size_t maxlen,
                          char* message) {

    std::string str = "";

    if (error == 0) {
        str += "Success.";
    }
    uint64_t code = static_cast<uint64_t>(error);

    //1. Extract TAPP (Bottom 5 bits)
    uint64_t tappVal = code & TAPP_FIELD_MASK;
    if (tappVal != 0) {
        str += " [TAPP Error]: ";
        switch (error)
        {
        case 1:
            str += "The extents for the indices shared between tensor A and B does not match.";
            break;
        case 2:
            str += "The extents for the indices shared between tensor A and D does not match.";
            break;
        case 3:
            str += "The extents for the indices shared between tensor B and D does not match.";
            break;
        case 4:
            str += "Tensor D has indices not shared with tensor A or B.";
            break;
        case 5:
            str += "The tensors C and D have different amount of dimensions.";
            break;
        case 6:
            str += "The indices of tensor C and D does not line up.";
            break;
        case 7:
            str += "The extents for the indices shared between tensor C and D does not match.";
            break;
        case 8:
            str += "Aliasing found within tensor D.";
            break;
        case 9:
            str += "An idx in tensor A has two different extents.";
            break;
        case 10:
            str += "An idx in tensor B has two different extents.";
            break;
        case 11:
            str += "An idx in tensor D has two different extents.";
            break;
        case 12:
            str += "C should not be NULL while beta is not zero.";
            break;
        case 13:
            str += "Nmode can not be negative.";
            break;
        case 14:
            str += "Extents can not be negative.";
            break;
        case 15:
            str += "Invalid attribute key.";
            break;
        default:
            str += "Unknown TAPP error code.";
            break;
        }
    }

    //2. Extract cuTENSOR (Middle 9 bits)
    uint64_t cutensorVal = (code & CUTENSOR_FIELD_MASK) >> CUTENSOR_OFFS;
    if (cutensorVal != 0) {
        cutensorStatus_t ts = static_cast<cutensorStatus_t>(cutensorVal);
        str += " [cuTENSOR Status]: ";
        str += cutensorGetErrorString(ts);
    }

    //3. Extract CUDA (Top 18 bits)
    int cudaVal = (code >> CUDA_OFFS);
    if (cudaVal != 0) {
        cudaError_t cs = static_cast<cudaError_t>(cudaVal);
        str += " [CUDA Error]: ";
        str += cudaGetErrorString(cs);
    }

    const char* error_message = str.c_str();
    size_t message_len = strlen(error_message);
    if (maxlen == 0) {
        return message_len;
    }
    size_t writelen = maxlen - 1 < message_len ? maxlen - 1 : message_len;
    strncpy(message, error_message, writelen);
    message[writelen] = '\0';
    return writelen;
}


int pack_error(int current_value, int tapp_err) {
    uint64_t val = static_cast<uint64_t>(current_value);
    uint64_t new_tapp_val = static_cast<uint64_t>(tapp_err);
    return static_cast<int>((val & TAPP_CLEAR_MASK) | new_tapp_val);
}

int pack_error(int current_value, cutensorStatus_t e) {
    uint64_t val = static_cast<uint64_t>(current_value);
    uint64_t new_tensor_val = static_cast<uint64_t>(e) << CUTENSOR_OFFS;
    return static_cast<int>((val & CUTENSOR_CLEAR_MASK) | new_tensor_val);
}

int pack_error(int current_value, cudaError_t e) {
    uint64_t val = static_cast<uint64_t>(current_value);
    uint64_t new_cuda_val = static_cast<uint64_t>(e) << CUDA_OFFS;
    uint64_t LOW_FIELDS_MASK = TAPP_FIELD_MASK | CUTENSOR_FIELD_MASK;
    uint64_t cleared_val = val & (~LOW_FIELDS_MASK);
    return static_cast<int>(cleared_val | new_cuda_val);
}
