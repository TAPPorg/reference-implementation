/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "tapp_ex_imp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "ctf-bind.h"

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides) {
    void* init_val;
    switch (type) { // tapp_datatype
    case TAPP_F32:
      float init_val_s = 0.0;
      init_val = &init_val_s;
      break;
    case TAPP_F64:
      double init_val_d = 0.0;
      init_val = &init_val_d;
      break;
    case TAPP_C32:
      float complex init_val_c = 0.0;
      init_val = &init_val_c;
      break;
    case TAPP_C64:
      double complex init_val_z = 0.0;
      init_val = &init_val_z;
      break;
    }
    return TAPP_create_tensor(info, type, nmode, extents, strides, init_val);
}


TAPP_error TAPP_create_tensor(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides, void* init_val) {
    if (nmode < 0) {
        return 14;
    }
    for (size_t i = 0; i < nmode; i++) {
        if (extents[i] < 0) {
            return 15;
        }
    }
    
    struct tensor_info* info_ptr = malloc(sizeof(struct tensor_info));

    info_ptr->type = type;
    info_ptr->nmode = nmode;

    info_ptr->extents = malloc(nmode * sizeof(int64_t));
    memcpy(info_ptr->extents, extents, nmode * sizeof(int64_t));

    info_ptr->strides = malloc(nmode * sizeof(int64_t));
    memcpy(info_ptr->strides, strides, nmode * sizeof(int64_t));

    info_ptr->uuid_len = distributed_get_uuid_len();

    info_ptr->uuid = malloc((info_ptr->uuid_len + 1) * sizeof(char));
    int ierr = distributed_get_uuid(info_ptr->uuid, info_ptr->uuid_len);

    *info = (TAPP_tensor_info)info_ptr;

    ierr = distributed_create_tensor(*info, init_val); //this should be made separate in a way that is appears as a call to a 2nd underlying tapp

    return 0;
}

TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    distributed_destruct_tensor(info);
    free(info_ptr->extents);
    free(info_ptr->strides);
    free(info_ptr->uuid);
    free(info_ptr);

    return 0;
}

int TAPP_get_nmodes(TAPP_tensor_info info) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    return info_ptr->nmode;
}

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           int nmodes) {
    if (nmodes < 0) {
        return 14;
    }
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    info_ptr->nmode = nmodes;
    info_ptr->extents = realloc(info_ptr->extents, info_ptr->nmode * sizeof(int64_t));
    info_ptr->strides = realloc(info_ptr->strides, info_ptr->nmode * sizeof(int64_t));
    
    return 0;
}

void TAPP_get_extents(TAPP_tensor_info info,
                      int64_t* extents) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    memcpy(extents, info_ptr->extents, info_ptr->nmode * sizeof(int64_t));
}

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            const int64_t* extents) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;

    for (size_t i = 0; i < info_ptr->nmode; i++)
    {
        if (extents[i] < 0) {
            return 15;
        }
    }
    
    memcpy(info_ptr->extents, extents, info_ptr->nmode * sizeof(int64_t));
    
    return 0;
}

void TAPP_get_strides(TAPP_tensor_info info,
                      int64_t* strides) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    memcpy(strides, info_ptr->strides, info_ptr->nmode * sizeof(int64_t));
}

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            const int64_t* strides) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    memcpy(info_ptr->strides, strides, info_ptr->nmode * sizeof(int64_t));
    
    return 0;
}

int TAPP_get_uuid_len(TAPP_tensor_info info) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    return info_ptr->uuid_len;
}

TAPP_error TAPP_get_uuid(TAPP_tensor_info info,
                      char * uuid, const int uuid_len) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    if(uuid_len != info_ptr->uuid_len) {
      printf("Error: wrong char buffer size in TAPP_get_uuid.\n");
      return 184;
    }
    memcpy(uuid, info_ptr->uuid, (info_ptr->uuid_len+1)*sizeof(char));
    return 0;
}

TAPP_datatype TAPP_get_datatype(TAPP_tensor_info info){
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    return info_ptr->type;
}

TAPP_error TAPP_broadcast_tensor_body_data(TAPP_tensor_info info, void* data){
    return distributed_broadcast_tensor_body_data(info, data);
}

TAPP_error TAPP_gather_tensor_body_data(TAPP_tensor_info info, void* data){
    return distributed_gather_tensor_body_data(info, data);
}

TAPP_error TAPP_tensor_set_name(TAPP_tensor_info info,
                      const int64_t* name, const int name_len){
    return distributed_tensor_set_name(info, name, name_len);
}

TAPP_error TAPP_initialize_tensor(TAPP_tensor_info info, void* init_val){
    return distributed_initialize_tensor(info, init_val);
}

TAPP_error TAPP_copy_tensor(TAPP_tensor_info dest, TAPP_tensor_info src){
    return distributed_copy_tensor(dest, src);
}

TAPP_error TAPP_scale_with_denominators(TAPP_tensor_info info,
                                 const int n_occ, const int n_vir, void* eps_occ, void* eps_vir, void* eps_ijk){
    return distributed_scale_with_denominators(info, n_occ, n_vir, eps_occ, eps_vir, eps_ijk);
}

TAPP_error TAPP_reshape_tensor(TAPP_tensor_info dest, TAPP_tensor_info src, int nmode, int64_t* dimensions){

    return distributed_reshape_tensor(dest, src, nmode, dimensions);
}

