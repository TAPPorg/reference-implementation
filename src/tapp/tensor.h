#ifndef TAPP_TENSOR_H_
#define TAPP_TENSOR_H_

#include <stdint.h>

#include "error.h"
#include "datatype.h"

typedef intptr_t TAPP_tensor_info;

/*
 * TODO: what are the required error conditions? What is explicitly allowed (esp. regarding strides)?
 */

/*
 * Niklas: Is there a reason to allow negative extents?
 *         Should it be error checked or just disallowed by
 *         using unsigned values? Same with nmode.
 */ 

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides);

TAPP_error TAPP_create_tensor(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides, void* init_val);

TAPP_error TAPP_initialize_tensor(TAPP_tensor_info info, void* init_val);

TAPP_error TAPP_destory_tensor_info(TAPP_tensor_info info);

int TAPP_get_nmodes(TAPP_tensor_info info);

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           int nmodes);

void TAPP_get_extents(TAPP_tensor_info info,
                      int64_t* extents);

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            const int64_t* extents);

void TAPP_get_strides(TAPP_tensor_info info,
                      int64_t* strides);

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            const int64_t* strides);

int TAPP_get_uuid_len(TAPP_tensor_info info); 

TAPP_error TAPP_get_uuid(TAPP_tensor_info info,
                      char * uuid, const int uuid_len);

TAPP_datatype TAPP_get_datatype(TAPP_tensor_info info);

TAPP_error TAPP_broadcast_tensor_body_data(TAPP_tensor_info info, void* data);

TAPP_error TAPP_gather_tensor_body_data(TAPP_tensor_info info, void* data);

TAPP_error TAPP_tensor_set_name(TAPP_tensor_info info,
                      const int64_t* name, const int name_len);

TAPP_error TAPP_copy_tensor(TAPP_tensor_info dest, TAPP_tensor_info src); 

TAPP_error TAPP_scale_with_denominators(TAPP_tensor_info info, 
		                  const int n_occ, const int n_vir, void* eps_occ, void* eps_vir, void* eps_ijk);

#endif /* TAPP_TENSOR_H_ */
