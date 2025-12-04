/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - November 2025
 */

#include "tapp_ex_imp.h"
#include <complex.h>

int64_t* sort_by_idx(int nmode, int64_t* list, int64_t* idx, int64_t* sorted_idx);
int64_t calculate_size(int64_t* extents, int nmode);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
int calculate_index(int nmode, int64_t* strides, int64_t* coordinates);
void calculate_alpha_X_beta_Y(TAPP_datatype type_Z, int64_t index_Z, void* Z, TAPP_datatype type_X, void* alpha, int64_t index_X, void* X, TAPP_datatype type_Y, void* beta, int64_t index_Y, void* Y);

TAPP_error TAPP_create_tensor_transpose(TAPP_tensor_transpose* plan,
                                        TAPP_handle handle,
                                        TAPP_element_op op_X,
                                        TAPP_tensor_info X,
                                        const int64_t* idx_X,
                                        TAPP_element_op op_Y,
                                        TAPP_tensor_info Y,
                                        const int64_t* idx_Y,
                                        TAPP_element_op op_Z,
                                        TAPP_tensor_info Z,
                                        const int64_t* idx_Z)
{
    // TODO: Should repeated indices be allowed? Would create ambiguous situations
    struct transpose_plan* plan_ptr = malloc(sizeof(struct transpose_plan));
    plan_ptr->handle = handle;

    plan_ptr->op_X = op_X;
    plan_ptr->X = X;

    plan_ptr->idx_X = malloc(((struct tensor_info*)X)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_X, idx_X, ((struct tensor_info*)X)->nmode * sizeof(int64_t));


    plan_ptr->op_Y = op_Y;
    plan_ptr->Y = Y;

    plan_ptr->idx_Y = malloc(((struct tensor_info*)Y)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_Y, idx_Y, ((struct tensor_info*)Y)->nmode * sizeof(int64_t));


    plan_ptr->op_Z = op_Z;
    plan_ptr->Z = Z;

    plan_ptr->idx_Z = malloc(((struct tensor_info*)Z)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_Z, idx_Z, ((struct tensor_info*)Z)->nmode * sizeof(int64_t));

    *plan = (TAPP_tensor_transpose)plan_ptr;

    return 0;
}

TAPP_error TAPP_destroy_tensor_transpose(TAPP_tensor_transpose plan)
{
    free(((struct transpose_plan*)plan)->idx_X);
    free(((struct transpose_plan*)plan)->idx_Y);
    free(((struct transpose_plan*)plan)->idx_Z);
    free((struct transpose_plan*)plan);

    return 0;
}

TAPP_error TAPP_execute_transpose(TAPP_tensor_transpose plan,
                                  TAPP_executor exec,
                                  TAPP_status* status,
                                  const void* alpha,
                                  const void* X,
                                  const void* Y,
                                  const void* beta,
                                        void* Z)
{
    struct transpose_plan* plan_ptr = (struct transpose_plan*)plan;
    TAPP_handle handle = plan_ptr->handle;

    TAPP_element_op op_X = plan_ptr->op_X;
    TAPP_tensor_info X_info = (TAPP_tensor_info)(plan_ptr->X);
    struct tensor_info* X_info_ptr = (struct tensor_info*)(plan_ptr->X);

    TAPP_element_op op_Y = plan_ptr->op_Y;
    TAPP_tensor_info Y_info = (TAPP_tensor_info)(plan_ptr->Y);
    struct tensor_info* Y_info_ptr = (struct tensor_info*)(plan_ptr->Y);

    TAPP_element_op op_Z = plan_ptr->op_Z;
    TAPP_tensor_info Z_info = (TAPP_tensor_info)(plan_ptr->Z);
    struct tensor_info* Z_info_ptr = (struct tensor_info*)(plan_ptr->Z);

    TAPP_datatype type_X = X_info_ptr->type;
    int nmode_X = TAPP_get_nmodes(X_info);
    int64_t* extents_X = malloc(nmode_X * sizeof(int64_t));
    TAPP_get_extents(X_info, extents_X);
    int64_t* strides_X = malloc(nmode_X * sizeof(int64_t));
    TAPP_get_strides(X_info, strides_X);
    int64_t* idx_X = malloc(nmode_X * sizeof(int64_t));
    memcpy(idx_X, plan_ptr->idx_X, nmode_X * sizeof(int64_t));

    TAPP_datatype type_Y = Y_info_ptr->type;
    int nmode_Y = TAPP_get_nmodes(Y_info);
    int64_t* extents_Y = malloc(nmode_Y * sizeof(int64_t));
    TAPP_get_extents(Y_info, extents_Y);
    int64_t* strides_Y = malloc(nmode_Y * sizeof(int64_t));
    TAPP_get_strides(Y_info, strides_Y);
    int64_t* idx_Y = malloc(nmode_Y * sizeof(int64_t));
    memcpy(idx_Y, plan_ptr->idx_Y, nmode_Y * sizeof(int64_t));

    TAPP_datatype type_Z = Z_info_ptr->type;
    int nmode_Z = TAPP_get_nmodes(Z_info);
    int64_t* extents_Z = malloc(nmode_Z * sizeof(int64_t));
    TAPP_get_extents(Z_info, extents_Z);
    int64_t* strides_Z = malloc(nmode_Z * sizeof(int64_t));
    TAPP_get_strides(Z_info, strides_Z);
    int64_t* idx_Z = malloc(nmode_Z * sizeof(int64_t));
    memcpy(idx_Z, plan_ptr->idx_Z, nmode_Z * sizeof(int64_t));

    int64_t* sorted_extents_X = sort_by_idx(nmode_X, extents_X, idx_X, idx_Z);
    int64_t* sorted_strides_X = sort_by_idx(nmode_X, strides_X, idx_X, idx_Z);
    int64_t* sorted_extents_Y = sort_by_idx(nmode_Y, extents_Y, idx_Y, idx_Z);
    int64_t* sorted_strides_Y = sort_by_idx(nmode_Y, strides_Y, idx_Y, idx_Z);

    int64_t size_Z = calculate_size(extents_Z, nmode_Z);

    int64_t* coordinates = malloc(nmode_Z * sizeof(int64_t));
    for (size_t i = 0; i < nmode_Z; i++)
    {
        coordinates[i] = 0;
    }    

    for (int64_t i = 0; i < size_Z; i++)
    {
        int index_X = calculate_index(nmode_X, sorted_strides_X, coordinates);
        int index_Y = calculate_index(nmode_Y, sorted_strides_Y, coordinates);
        int index_Z = calculate_index(nmode_Z, strides_Z, coordinates);
        calculate_alpha_X_beta_Y(type_Z, index_Z, Z, type_X, alpha, index_X, X, type_Y, beta, index_Y, Y);
        increment_coordinates(coordinates, nmode_Z, extents_Z);
    }
    
}

int64_t* sort_by_idx(int nmode, int64_t* list, int64_t* idx, int64_t* sorted_idx)
{
    int64_t* sorted_list = malloc(nmode * sizeof(int64_t));
    for (size_t i = 0; i < nmode; i++)
    {
        for (size_t j = 0; j < nmode; j++)
        {
            if (idx[i] == sorted_idx[j])
            {
                sorted_list[j] = list[i];
                break;
            }
        }
    }
    
    return sorted_list;
}

int64_t calculate_size(int64_t* extents, int nmode)
{
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    return size;
}

void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents)
{
    if (nmode <= 0)
    {
        return;
    }

    int k = 0;
    do
    {
        coordinates[k] = (coordinates[k] + 1) % extents[k];
        k++;
    } while (coordinates[k - 1] == 0 && k < nmode);
}

int calculate_index(int nmode, int64_t* strides, int64_t* coordinates)
{
    int index = 0;
    for (int i = 0; i < nmode; i++)
    {
        index += coordinates[i] * strides[i];
    }
    return index;
}

void calculate_alpha_X_beta_Y(TAPP_datatype type_Z, int64_t index_Z, void* Z, TAPP_datatype type_X, void* alpha, int64_t index_X, void* X, TAPP_datatype type_Y, void* beta, int64_t index_Y, void* Y)
{
    switch (type_Z)
    {
    case TAPP_F32:
        switch (type_X)
        {
        case TAPP_F32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((float*)Z)[index_Z] = (float)(*((float*)alpha) * ((float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((float*)Z)[index_Z] = (float)(*((float*)alpha) * ((float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((float*)Z)[index_Z] = (float)(*((float*)alpha) * ((float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((float*)Z)[index_Z] = (float)(*((float*)alpha) * ((float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((float*)Z)[index_Z] = (float)(*((float*)alpha) * ((float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((float*)Z)[index_Z] = (float)(*((float*)alpha) * ((float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((float*)Z)[index_Z] = (float)(*((double*)alpha) * ((double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((float*)Z)[index_Z] = (float)(*((double*)alpha) * ((double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((float*)Z)[index_Z] = (float)(*((double*)alpha) * ((double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((float*)Z)[index_Z] = (float)(*((double*)alpha) * ((double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((float*)Z)[index_Z] = (float)(*((double*)alpha) * ((double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((float*)Z)[index_Z] = (float)(*((double*)alpha) * ((double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((float*)Z)[index_Z] = (float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((float*)Z)[index_Z] = (float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((float*)Z)[index_Z] = (float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((float*)Z)[index_Z] = (float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((float*)Z)[index_Z] = (float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((float*)Z)[index_Z] = (float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((float*)Z)[index_Z] = (float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((float*)Z)[index_Z] = (float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((float*)Z)[index_Z] = (float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((float*)Z)[index_Z] = (float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((float*)Z)[index_Z] = (float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((float*)Z)[index_Z] = (float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef ENABLE_F16
        case TAPP_F16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((float*)Z)[index_Z] = (float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((float*)Z)[index_Z] = (float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((float*)Z)[index_Z] = (float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((float*)Z)[index_Z] = (float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((float*)Z)[index_Z] = (float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((float*)Z)[index_Z] = (float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef ENABLE_FB16
        case TAPP_BF16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((float*)Z)[index_Z] = (float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((float*)Z)[index_Z] = (float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((float*)Z)[index_Z] = (float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((float*)Z)[index_Z] = (float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((float*)Z)[index_Z] = (float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((float*)Z)[index_Z] = (float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F64:
        switch (type_X)
        {
        case TAPP_F32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((double*)Z)[index_Z] = (double)(*((float*)alpha) * ((float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((double*)Z)[index_Z] = (double)(*((float*)alpha) * ((float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((double*)Z)[index_Z] = (double)(*((float*)alpha) * ((float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((double*)Z)[index_Z] = (double)(*((float*)alpha) * ((float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((double*)Z)[index_Z] = (double)(*((float*)alpha) * ((float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((double*)Z)[index_Z] = (double)(*((float*)alpha) * ((float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((double*)Z)[index_Z] = (double)(*((double*)alpha) * ((double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((double*)Z)[index_Z] = (double)(*((double*)alpha) * ((double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((double*)Z)[index_Z] = (double)(*((double*)alpha) * ((double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((double*)Z)[index_Z] = (double)(*((double*)alpha) * ((double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((double*)Z)[index_Z] = (double)(*((double*)alpha) * ((double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((double*)Z)[index_Z] = (double)(*((double*)alpha) * ((double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((double*)Z)[index_Z] = (double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((double*)Z)[index_Z] = (double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((double*)Z)[index_Z] = (double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((double*)Z)[index_Z] = (double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((double*)Z)[index_Z] = (double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((double*)Z)[index_Z] = (double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((double*)Z)[index_Z] = (double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((double*)Z)[index_Z] = (double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((double*)Z)[index_Z] = (double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((double*)Z)[index_Z] = (double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((double*)Z)[index_Z] = (double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((double*)Z)[index_Z] = (double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef ENABLE_F16
        case TAPP_F16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((double*)Z)[index_Z] = (double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((double*)Z)[index_Z] = (double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((double*)Z)[index_Z] = (double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((double*)Z)[index_Z] = (double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((double*)Z)[index_Z] = (double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((double*)Z)[index_Z] = (double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef ENABLE_FB16
        case TAPP_BF16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((double*)Z)[index_Z] = (double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((double*)Z)[index_Z] = (double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((double*)Z)[index_Z] = (double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((double*)Z)[index_Z] = (double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((double*)Z)[index_Z] = (double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((double*)Z)[index_Z] = (double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_C32:
        switch (type_X)
        {
        case TAPP_F32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex float*)Z)[index_Z] = (complex float)(*((float*)alpha) * ((float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex float*)Z)[index_Z] = (complex float)(*((float*)alpha) * ((float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex float*)Z)[index_Z] = (complex float)(*((float*)alpha) * ((float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex float*)Z)[index_Z] = (complex float)(*((float*)alpha) * ((float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex float*)Z)[index_Z] = (complex float)(*((float*)alpha) * ((float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex float*)Z)[index_Z] = (complex float)(*((float*)alpha) * ((float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex float*)Z)[index_Z] = (complex float)(*((double*)alpha) * ((double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex float*)Z)[index_Z] = (complex float)(*((double*)alpha) * ((double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex float*)Z)[index_Z] = (complex float)(*((double*)alpha) * ((double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex float*)Z)[index_Z] = (complex float)(*((double*)alpha) * ((double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex float*)Z)[index_Z] = (complex float)(*((double*)alpha) * ((double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex float*)Z)[index_Z] = (complex float)(*((double*)alpha) * ((double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex float*)Z)[index_Z] = (complex float)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef ENABLE_F16
        case TAPP_F16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex float*)Z)[index_Z] = (complex float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex float*)Z)[index_Z] = (complex float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex float*)Z)[index_Z] = (complex float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex float*)Z)[index_Z] = (complex float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex float*)Z)[index_Z] = (complex float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex float*)Z)[index_Z] = (complex float)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef ENABLE_FB16
        case TAPP_BF16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex float*)Z)[index_Z] = (complex float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex float*)Z)[index_Z] = (complex float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex float*)Z)[index_Z] = (complex float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex float*)Z)[index_Z] = (complex float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex float*)Z)[index_Z] = (complex float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex float*)Z)[index_Z] = (complex float)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_C64:
        switch (type_X)
        {
        case TAPP_F32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex double*)Z)[index_Z] = (complex double)(*((float*)alpha) * ((float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex double*)Z)[index_Z] = (complex double)(*((float*)alpha) * ((float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex double*)Z)[index_Z] = (complex double)(*((float*)alpha) * ((float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex double*)Z)[index_Z] = (complex double)(*((float*)alpha) * ((float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex double*)Z)[index_Z] = (complex double)(*((float*)alpha) * ((float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex double*)Z)[index_Z] = (complex double)(*((float*)alpha) * ((float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex double*)Z)[index_Z] = (complex double)(*((double*)alpha) * ((double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex double*)Z)[index_Z] = (complex double)(*((double*)alpha) * ((double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex double*)Z)[index_Z] = (complex double)(*((double*)alpha) * ((double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex double*)Z)[index_Z] = (complex double)(*((double*)alpha) * ((double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex double*)Z)[index_Z] = (complex double)(*((double*)alpha) * ((double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex double*)Z)[index_Z] = (complex double)(*((double*)alpha) * ((double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex double*)Z)[index_Z] = (complex double)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef ENABLE_F16
        case TAPP_F16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex double*)Z)[index_Z] = (complex double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex double*)Z)[index_Z] = (complex double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex double*)Z)[index_Z] = (complex double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex double*)Z)[index_Z] = (complex double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex double*)Z)[index_Z] = (complex double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex double*)Z)[index_Z] = (complex double)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef ENABLE_FB16
        case TAPP_BF16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((complex double*)Z)[index_Z] = (complex double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((complex double*)Z)[index_Z] = (complex double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((complex double*)Z)[index_Z] = (complex double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((complex double*)Z)[index_Z] = (complex double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((complex double*)Z)[index_Z] = (complex double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((complex double*)Z)[index_Z] = (complex double)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
        default:
            break;
        }
        break;
#ifdef ENABLE_F16
    case TAPP_F16:
        switch (type_X)
        {
        case TAPP_F32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((float*)alpha) * ((float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((float*)alpha) * ((float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((float*)alpha) * ((float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((float*)alpha) * ((float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((float*)alpha) * ((float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((float*)alpha) * ((float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((double*)alpha) * ((double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((double*)alpha) * ((double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((double*)alpha) * ((double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((double*)alpha) * ((double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((double*)alpha) * ((double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((double*)alpha) * ((double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef ENABLE_F16
        case TAPP_F16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef ENABLE_FB16
        case TAPP_BF16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((_Float16*)Z)[index_Z] = (_Float16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
        default:
            break;
        }
        break;
#endif
#ifdef ENABLE_BF16
    case TAPP_BF16:
        switch (type_X)
        {
        case TAPP_F32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((float*)alpha) * ((float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((float*)alpha) * ((float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((float*)alpha) * ((float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((float*)alpha) * ((float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((float*)alpha) * ((float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((float*)alpha) * ((float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((double*)alpha) * ((double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((double*)alpha) * ((double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((double*)alpha) * ((double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((double*)alpha) * ((double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((double*)alpha) * ((double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((double*)alpha) * ((double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_Y)
            {
            case TAPP_F32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex float*)alpha) * ((complex float*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_Y)
            {
            case TAPP_F32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((complex double*)alpha) * ((complex double*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef ENABLE_F16
        case TAPP_F16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((_Float16*)alpha) * ((_Float16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef ENABLE_FB16
        case TAPP_BF16:
            switch (type_Y)
            {
            case TAPP_F32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((float*)beta) * ((float*)Y)[index_Y]);
                break;
            case TAPP_F64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((double*)beta) * ((double*)Y)[index_Y]);
                break;
            case TAPP_C32:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex float*)beta) * ((complex float*)Y)[index_Y]);
                break;
            case TAPP_C64:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((complex double*)beta) * ((complex double*)Y)[index_Y]);
                break;
#ifdef ENABLE_F16
            case TAPP_F16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((_Float16*)beta) * ((_Float16*)Y)[index_Y]);
                break;
#endif
#ifdef ENABLE_BF16
            case TAPP_BF16:
                ((__bf16*)Z)[index_Z] = (__bf16)(*((__bf16*)alpha) * ((__bf16*)X)[index_X] + *((__bf16*)beta) * ((__bf16*)Y)[index_Y]);
                break;
#endif
            default:
                break;
            }
            break;
#endif
        default:
            break;
        }
        break;
#endif
    default:
        break;
    }
}