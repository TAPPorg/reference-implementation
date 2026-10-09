#ifndef TAPP_TBLIS_PRODUCT_H_
#define TAPP_TBLIS_PRODUCT_H_

#include <tapp.h>

#include <vector>

#include "tblis/frame/base/basic_types.h"
#include "tblis_utils.h"
#include "tensor.h"

struct product_plan
{
    tensor_info* A;
    tensor_info* B;
    tensor_info* C;
    int64_t size_D;
    bool conj_scalars;
    std::vector<tblis::label_type> idx_A;
    std::vector<tblis::label_type> idx_B;
    std::vector<tblis::label_type> idx_C;
    tblis::type_t scalar_type;
};

bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B);

TAPP_error execute_product(const product_plan* plan,
                           const tblis::tblis_scalar alpha,
                           const void* A,
                           const void* B,
                           const tblis::tblis_scalar beta,
                           const void* C,
                                 void* D);

#endif  /* TAPP_TBLIS_PRODUCT_H_ */
