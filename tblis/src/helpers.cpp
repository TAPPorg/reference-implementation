#include "internal.h"

bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B) {
    if (A->extents != B->extents || A->strides != B->strides) return false;
    for (size_t i = 0; i < A->extents.size(); i++) {
        if (idx_A[i] != idx_B[i]) return false;
    }
    return true;
}