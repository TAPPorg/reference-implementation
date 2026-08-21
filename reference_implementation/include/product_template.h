#ifndef TAPP_REF_IMPL_REF_IMPL_PRODUCT_TEMPLATE_H_
#define TAPP_REF_IMPL_REF_IMPL_PRODUCT_TEMPLATE_H_

#include "product.h"

int64_t calcualte_offset(int64_t* coords, int nmode, int64_t* strides);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
int check_executor_existence(TAPP_executor exec, int error_code);

#endif  /* TAPP_REF_IMPL_REF_IMPL_PRODUCT_TEMPLATE_H_ */
