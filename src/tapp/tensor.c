/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "tapp_ex_imp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides) {
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

    *info = (TAPP_tensor_info)info_ptr;

    return 0;
}

TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info) {
    struct tensor_info* info_ptr = (struct tensor_info*)info;
    free(info_ptr->extents);
    free(info_ptr->strides);
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