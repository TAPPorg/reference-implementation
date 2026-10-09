/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "../include/tblis_utils.h"

#include <limits>

bool translate_datatype_to_tblis(TAPP_datatype type, tblis::type_t* tblis_type) {
    // The numeric values differ between TAPP and TBLIS (e.g. TAPP_F64 = 1 but TYPE_DOUBLE = 2), so no casting
    switch (type) {
    case TAPP_F32:
        *tblis_type = tblis::TYPE_FLOAT;
        return true;
    case TAPP_F64:
        *tblis_type = tblis::TYPE_DOUBLE;
        return true;
    case TAPP_C32:
        *tblis_type = tblis::TYPE_SCOMPLEX;
        return true;
    case TAPP_C64:
        *tblis_type = tblis::TYPE_DCOMPLEX;
        return true;
    default:
        return false;
    }
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

void init_tblis_tensor(tblis::tblis_tensor* tensor,
                       TAPP_datatype type,
                       int nmode,
                       const tblis::len_type* extents,
                       const tblis::stride_type* strides,
                       const void* data,
                       TAPP_element_op op,
                       const void* scalar) {
    // TODO: dispatch to tblis_init_tensor(_scaled)_{s,d,c,z}, set tensor->conj from op
}
