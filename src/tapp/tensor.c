/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides) {
    TAPP_datatype* type_ptr = malloc(sizeof(int64_t));
    *type_ptr = type;
    int* nmode_ptr = malloc(sizeof(int64_t));
    *nmode_ptr = nmode;

    intptr_t* info_ptr = malloc(4 * sizeof(intptr_t));
    info_ptr[0] = (intptr_t)type_ptr;
    info_ptr[1] = (intptr_t)nmode_ptr;
    info_ptr[2] = (intptr_t)extents;
    info_ptr[3] = (intptr_t)strides;

    *info = (TAPP_tensor_info)info_ptr;
}

TAPP_error TAPP_destory_tensor_info(TAPP_tensor_info info) {
    intptr_t* info_ptr = (intptr_t*)info;
    int64_t* type_ptr = (int64_t*)(info_ptr)[0];
    int64_t* nmode_ptr = (int64_t*)(info_ptr)[1];
    free(type_ptr);
    free(nmode_ptr);
    free(info_ptr);
}

int TAPP_get_nmodes(TAPP_tensor_info info) {
    int* nmode_ptr = (int*)((intptr_t*)info)[1];
    return *nmode_ptr;
}

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           int nmodes) {
    int* nmode_ptr = (int*)((intptr_t*)info)[1];
    *nmode_ptr = (int)nmodes;
}

void TAPP_get_extents(TAPP_tensor_info info,
                      int64_t* extents) {
    int64_t* extents_ptr = (int64_t*)((intptr_t*)info)[2];
    int nmodes = TAPP_get_nmodes(info);
    for (int i = 0; i < nmodes; i++) {
        extents[i] = extents_ptr[i];
    }
}

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            const int64_t* extents) {
    int64_t* extents_ptr = (int64_t*)((intptr_t*)info)[2];
    int nmodes = TAPP_get_nmodes(info);
    for (int i = 0; i < nmodes; i++) {
        extents_ptr[i] = extents[i];
    }
}

void TAPP_get_strides(TAPP_tensor_info info,
                      int64_t* strides) {
    int64_t* strides_ptr = (int64_t*)((intptr_t*)info)[3];
    int nmodes = TAPP_get_nmodes(info);
    for (int i = 0; i < nmodes; i++) {
        strides[i] = strides_ptr[i];
    }
}

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            const int64_t* strides) {
    int64_t* strides_ptr = (int64_t*)((intptr_t*)info)[3];
    int nmodes = TAPP_get_nmodes(info);
    for (int i = 0; i < nmodes; i++) {
        strides_ptr[i] = strides[i];
    }
}