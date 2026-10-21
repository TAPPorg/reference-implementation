/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "internal.h"

#include <limits>

bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B) {
    if (A->extents != B->extents || A->strides != B->strides) return false;
    for (size_t i = 0; i < A->extents.size(); i++) {
        if (idx_A[i] != idx_B[i]) return false;
    }
    return true;
}

bool translate_idx_to_tblis(int nmode,
                            const int64_t* idx,
                            std::unordered_map<int64_t, tblis::label_type>& label_map,
                            std::vector<tblis::label_type>& tblis_idx) {
    tblis_idx.resize(nmode);
    for (int i = 0; i < nmode; i++) {
        auto it = label_map.find(idx[i]);
        if (it == label_map.end()) {
            // Labels are handed out as 1, 2, 3, ... so 0 ('\0') is never used
            if (label_map.size() >= (size_t)std::numeric_limits<tblis::label_type>::max()) {
                return false;
            }
            it = label_map.emplace(idx[i], (tblis::label_type)(label_map.size() + 1)).first;
        }
        tblis_idx[i] = it->second;
    }
    return true;
}
