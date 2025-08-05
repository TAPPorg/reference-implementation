#ifndef HI_TAPP_TENSOR_H_
#define HI_TAPP_TENSOR_H_

#include <stdint.h>

#include "hi_tapp/error.h"
#include "hi_tapp/datatype.h"

typedef intptr_t HI_TAPP_tensor_info;

/*
 * TODO: what are the required error conditions? What is explicitly allowed (esp. regarding strides)?
 */

/*
 * Niklas: Is there a reason to allow negative extents?
 *         Should it be error checked or just disallowed by
 *         using unsigned values? Same with nmode.
 */ 

HI_TAPP_error HI_TAPP_create_tensor_info(HI_TAPP_tensor_info* info,
                                   HI_TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides);

HI_TAPP_error HI_TAPP_destory_tensor_info(HI_TAPP_tensor_info info);

int HI_TAPP_get_nmodes(HI_TAPP_tensor_info info);

HI_TAPP_error HI_TAPP_set_nmodes(HI_TAPP_tensor_info info,
                           int nmodes);

void HI_TAPP_get_extents(HI_TAPP_tensor_info info,
                      int64_t* extents);

HI_TAPP_error HI_TAPP_set_extents(HI_TAPP_tensor_info info,
                            const int64_t* extents);

void HI_TAPP_get_strides(HI_TAPP_tensor_info info,
                      int64_t* strides);

HI_TAPP_error HI_TAPP_set_strides(HI_TAPP_tensor_info info,
                            const int64_t* strides);

#endif /* HI_TAPP_TENSOR_H_ */
