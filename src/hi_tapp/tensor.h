#ifndef HI_TAPP_TENSOR_H_
#define HI_TAPP_TENSOR_H_

#include <stdint.h>

#include "hi_tapp/error.h"
#include "hi_tapp/datatype.h"
#include "hi_tapp/space.h"

typedef intptr_t HI_TAPP_tensor_info;

// two constructors of tensor_info, one for dense and one for block_sparse tensors


HI_TAPP_error HI_TAPP_create_tensor_info(HI_TAPP_tensor_info* info,
                            HI_TAPP_datatype type,
                            int nmode,
                            const int64_t* extents,
                            const int64_t* strides);

HI_TAPP_error HI_TAPP_create_block_sparse_tensor_info(HI_TAPP_tensor_info* info,
                            HI_TAPP_datatype type,
                            int nmode,
                            const HI_TAPP_space* spaces,
                            int64_t nblocks,
                            const int64_t* block_coordinates); //indices of sectors: nblocks x nmode, row-major

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

HI_TAPP_error HI_TAPP_get_nblocks(HI_TAPP_tensor_info info, 
                            int64_t* nblocks);

HI_TAPP_error HI_TAPP_set_nblocks(HI_TAPP_tensor_info info,
                            int64_t nblocks);



#endif /* HI_TAPP_TENSOR_H_ */
