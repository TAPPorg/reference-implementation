/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "../hi_tapp.h"
#include <stdint.h>

#ifdef __cplusplus

#include <vector>
#include <string>
#include <unordered_map>

struct VectorHasher;

struct space
{
    int64_t nsectors;
    int64_t* extents;
    int64_t nlabels; 
    HI_TAPP_attr* labels_names_and_extents;
    std::unordered_map<std::vector<int>, int, VectorHasher>* labels_to_sector_index;
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct tensor_info
{
    HI_TAPP_datatype type;
    int nmode;
    int64_t* extents; // NULL in block-sparse case
    int64_t* strides; // NULL in block-sparse case
    HI_TAPP_space* spaces; // NULL in dense case ? or 
    int64_t nblocks; // 0 in dense case ? or 1 ?
    int64_t* block_coordinates; // NULL in dense case ? or 1 row?
};

struct plan
{
    HI_TAPP_handle handle;
    HI_TAPP_element_op op_A;
    HI_TAPP_tensor_info A;
    int64_t* idx_A;
    HI_TAPP_element_op op_B;
    HI_TAPP_tensor_info B;
    int64_t* idx_B;
    HI_TAPP_element_op op_C;
    HI_TAPP_tensor_info C;
    int64_t* idx_C;
    HI_TAPP_element_op op_D;
    HI_TAPP_tensor_info D;
    int64_t* idx_D;
    HI_TAPP_prectype prec;
};

HI_TAPP_error create_executor(HI_TAPP_executor* exec);
HI_TAPP_error create_handle(HI_TAPP_handle* handle);

#ifdef __cplusplus
}
#endif
