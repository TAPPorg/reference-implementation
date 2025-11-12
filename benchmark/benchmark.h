#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "tapp.h"
#include <dlfcn.h>  // POSIX dynamic loading, TODO: fix for windows

#define NUMBER_OF_TESTS 48
#define TENSOR_A 0
#define TENSOR_B 1
#define TENSOR_D 2
#define NUMBER_OF_TENSORS 3

struct imp
{
    void* handle;
    TAPP_error (*TAPP_attr_set)(TAPP_attr attr, TAPP_key key, void* value);
    TAPP_error (*TAPP_attr_get)(TAPP_attr attr, TAPP_key key, void** value);
    TAPP_error (*TAPP_attr_clear)(TAPP_attr attr, TAPP_key key);
    bool (*TAPP_check_success)(TAPP_error error);
    size_t (*TAPP_explain_error)(TAPP_error error, size_t maxlen, char* message);
    TAPP_error (*create_executor)(TAPP_executor* exec);
    TAPP_error (*TAPP_destroy_executor)(TAPP_executor exec);
    TAPP_error (*create_handle)(TAPP_handle* handle);
    TAPP_error (*TAPP_destroy_handle)(TAPP_handle handle);
    TAPP_error (*TAPP_create_tensor_product)(TAPP_tensor_product* plan,
                                             TAPP_handle handle,
                                             TAPP_element_op op_A,
                                             TAPP_tensor_info A,
                                             const int64_t* idx_A,
                                             TAPP_element_op op_B,
                                             TAPP_tensor_info B,
                                             const int64_t* idx_B,
                                             TAPP_element_op op_C,
                                             TAPP_tensor_info C,
                                             const int64_t* idx_C,
                                             TAPP_element_op op_D,
                                             TAPP_tensor_info D,
                                             const int64_t* idx_D,
                                             TAPP_prectype prec);
    TAPP_error (*TAPP_destroy_tensor_product)(TAPP_tensor_product plan);
    TAPP_error (*TAPP_execute_product)(TAPP_tensor_product plan,
                                       TAPP_executor exec,
                                       TAPP_status* status,
                                       const void* alpha,
                                       const void* A,
                                       const void* B,
                                       const void* beta,
                                       const void* C,
                                             void* D);
    TAPP_error (*TAPP_execute_batched_product)(TAPP_tensor_product plan,
                                               TAPP_executor exec,
                                               TAPP_status* status,
                                               int num_batches,
                                               const void* alpha,
                                               const void** A,
                                               const void** B,
                                               const void* beta,
                                               const void** C,
                                                     void** D);
    TAPP_error (*TAPP_destroy_status)(TAPP_status status);
    TAPP_error (*TAPP_create_tensor_info)(TAPP_tensor_info* info,
                                          TAPP_datatype type,
                                          int nmode,
                                          const int64_t* extents,
                                          const int64_t* strides);
    TAPP_error (*TAPP_destroy_tensor_info)(TAPP_tensor_info info);
    int (*TAPP_get_nmodes)(TAPP_tensor_info info);
    TAPP_error (*TAPP_set_nmodes)(TAPP_tensor_info info, int nmodes);
    void (*TAPP_get_extents)(TAPP_tensor_info info, int64_t* extents);
    TAPP_error (*TAPP_set_extents)(TAPP_tensor_info info, const int64_t* extents);
    void (*TAPP_get_strides)(TAPP_tensor_info info, int64_t* strides);
    TAPP_error (*TAPP_set_strides)(TAPP_tensor_info info, const int64_t* strides);
};

char* imp_path = "./lib/libcutensor_binds.so";


const char* indices_list[NUMBER_OF_TESTS] = {
    "abc-bda-dc",
    "abc-dca-bd",
    "abcd-dbea-ec",
    "abcd-deca-be",
    "abcd-ebad-ce",
    "abcde-efbad-cf",
    "abcde-ecbfa-fd",
    "abcde-efcad-bf",
    "abcd-ea-ebcd",
    "abcd-eb-aecd",
    "abcd-ec-abed",
    "ab-ac-cb",
    "ab-acd-dbc",
    "ab-cad-dcb",
    "abc-acd-db",
    "abc-ad-bdc",
    "abc-adc-bd",
    "abc-adc-db",
    "abc-adec-ebd",
    "abcd-aebf-dfce",
    "abcd-aebf-fdec",
    "abcd-aecf-bfde",
    "abcd-aecf-fbed",
    "abcd-aedf-bfce",
    "abcd-aedf-fbec",
    "abcd-aefb-fdce",
    "abcd-aefc-fbed",
    "abcd-eafb-fdec",
    "abcd-eafc-bfde",
    "abcd-eafd-fbec",
    "abcdef-dega-gfbc",
    "abcdef-degb-gfac",
    "abcdef-degc-gfab",
    "abcdef-dfga-gebc",
    "abcdef-dfgb-geac",
    "abcdef-dfgc-geab",
    "abcdef-efga-gdbc",
    "abcdef-efgb-gdac",
    "abcdef-efgc-gdab",
    "abcdef-gdab-efgc",
    "abcdef-gdac-efgb",
    "abcdef-gdbc-efga",
    "abcdef-geab-dfgc",
    "abcdef-geac-dfgb",
    "abcdef-gebc-dfga",
    "abcdef-gfab-degc",
    "abcdef-gfac-degb",
    "abcdef-gfbc-dega",
};
const int64_t extents_list[NUMBER_OF_TESTS][7] = { // Extents in alphabetic order for the indices starting on 'a'
    {384,384,24,384},
    {384,24,376,384},
    {96,84,24,96,96},
    {96,24,84,96,84},
    {96,84,24,84,96},
    {48,36,24,36,48,36},
    {48,36,36,24,48,48},
    {48,24,36,36,48,36},
    {96,84,84,84,96},
    {96,84,84,84,96},
    {96,84,84,84,96},
    {7248,7240,7248},
    {384,376,376,384},
    {384,376,384,384},
    {384,376,376,384},
    {384,384,376,376},
    {384,384,376,376},
    {384,376,376,384},
    {96,84,84,84,96},
    {96,84,84,96,84,84},
    {96,84,84,84,84,96},
    {96,96,84,84,84,84},
    {96,84,84,84,84,96},
    {96,96,84,84,84,84},
    {96,84,84,84,84,96},
    {96,84,84,84,84,96},
    {96,84,84,84,84,96},
    {96,84,84,84,96,96},
    {96,96,84,84,96,84},
    {96,84,84,84,96,96},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,20,24,20,24},
    {24,20,20,20,24,20,24},
    {24,20,20,20,24,20,24},
    {24,20,20,20,24,20,24},
    {24,20,20,20,24,20,24},
    {24,20,20,20,24,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24},
    {24,20,20,24,20,20,24}
};

struct imp imp;