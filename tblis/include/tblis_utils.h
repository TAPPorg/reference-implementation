#ifndef TAPP_TBLIS_TBLIS_UTILS_H_
#define TAPP_TBLIS_TBLIS_UTILS_H_

#pragma GCC diagnostic push
#include "tblis.h"
#pragma GCC diagnostic pop

#include <tapp.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

// Maps a TAPP storage datatype to the corresponding TBLIS type, returns false if unsupported (e.g. F16, BF16)
bool translate_datatype_to_tblis(TAPP_datatype type, tblis::type_t* tblis_type);

// Maps TAPP index labels to TBLIS labels (char). label_map must be shared between all tensors of one product so that
// equal TAPP labels get equal TBLIS labels. Returns false if the product uses more distinct labels than TBLIS supports.
bool translate_idx_to_tblis(int nmode,
                            const int64_t* idx,
                            std::unordered_map<int64_t, tblis::label_type>& label_map,
                            std::vector<tblis::label_type>& tblis_idx);

// Wraps raw data in a tblis_tensor, optionally scaled (alpha/beta) and conjugated (op)
void init_tblis_tensor(tblis::tblis_tensor* tensor,
                       TAPP_datatype type,
                       int nmode,
                       const tblis::len_type* extents,
                       const tblis::stride_type* strides,
                       const void* data,
                       TAPP_element_op op,
                       const void* scalar);

#endif  /* TAPP_TBLIS_TBLIS_UTILS_H_ */
