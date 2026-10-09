#ifndef TAPP_TBLIS_TENSOR_H_
#define TAPP_TBLIS_TENSOR_H_

#include <tapp.h>

#include <vector>

#include "tblis/frame/base/basic_types.h"
#include "tblis_utils.h"

// Owns its extents and strides arrays (allocated with new[], freed in TAPP_destroy_tensor_info)
struct tensor_info
{
    tblis::type_t type;
    int nmode;
    tblis::len_type* extents;
    tblis::stride_type* strides;
    int conj;
};

// Deep copy of info, including the extents and strides arrays.
// Returns 0 if info is 0 or allocation fails. Free with TAPP_destroy_tensor_info.
tensor_info* clone_tensor_info(tensor_info* info);
TAPP_error destroy_tensor_info(tensor_info* info);

// Creates a tblis_tensor with its own copy of the shape of info, data is left as nullptr.
// Returns nullptr if info is null, its datatype is unsupported, or allocation fails. Free with free_tensor.
tblis::tblis_tensor* create_tblis_tensor(const tensor_info* info);

void free_tblis_tensor(tblis::tblis_tensor* t);

#endif  /* TAPP_TBLIS_TENSOR_H_ */
