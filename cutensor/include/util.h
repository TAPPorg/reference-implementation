#ifndef UTIL_H_
#define UTIL_H_

#include <cutensor.h>

#include <tapp.h>

const uint32_t kAlignment = 128;
int64_t compute_index(const int64_t* coordinates, int nmode, const int64_t* strides);
void increment_coordinates(int64_t* coordinates, int nmode, const int64_t* extents);
cutensorOperator_t translate_operator(TAPP_element_op op);

#endif
