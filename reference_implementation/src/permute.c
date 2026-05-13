#include "../include/permute.h"

TAPP_error TAPP_create_tensor_permute(TAPP_tensor_permute* plan,
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
                                      TAPP_prectype prec)
{
    struct tensor_info* info_A_ptr = (struct tensor_info*)A;
    struct tensor_info* info_B_ptr = (struct tensor_info*)B;
    struct tensor_info* info_C_ptr = (struct tensor_info*)C;

    struct plan* plan_ptr = malloc(sizeof(struct plan));

    plan_ptr->type_A = info_A_ptr->type;
    plan_ptr->type_B = info_B_ptr->type;
    plan_ptr->type_C = info_C_ptr->type;

    plan_ptr->op_A = op_A;
    plan_ptr->op_B = op_B;
    plan_ptr->op_C = op_C;
    plan_ptr->prec = prec;
    plan_ptr->I_nmode = extract_IX_indices(((struct tensor_info*)A)->nmode, idx_A,
                                           ((struct tensor_info*)C)->nmode, idx_C,
                                           &plan_ptr->I_idx);
    plan_ptr->B_nmode = extract_IX_indices(((struct tensor_info*)C)->nmode, idx_C,
                                           ((struct tensor_info*)A)->nmode, idx_A,
                                           &plan_ptr->B_idx);
    plan_ptr->P_nmode = extract_P_indices(((struct tensor_info*)A)->nmode, idx_A,
                                          ((struct tensor_info*)C)->nmode, idx_C,
                                          &plan_ptr->P_idx);

    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->I_nmode, plan_ptr->I_idx, &plan_ptr->I_extents);
    extract_grouped_extents(((struct tensor_info*)C)->nmode, idx_C, ((struct tensor_info*)C)->extents, plan_ptr->B_nmode, plan_ptr->B_idx, &plan_ptr->B_extents);
    extract_grouped_extents(((struct tensor_info*)C)->nmode, idx_C, ((struct tensor_info*)C)->extents, plan_ptr->P_nmode, plan_ptr->P_idx, &plan_ptr->P_extents);
    
    extract_grouped_strides(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->strides, plan_ptr->I_nmode, plan_ptr->I_idx, &plan_ptr->I_strides);
    extract_grouped_strides(((struct tensor_info*)C)->nmode, idx_C, ((struct tensor_info*)C)->strides, plan_ptr->B_nmode, plan_ptr->B_idx, &plan_ptr->B_strides);
    extract_grouped_strides(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->strides, plan_ptr->P_nmode, plan_ptr->P_idx, &plan_ptr->P_strides_A);
    extract_grouped_strides(((struct tensor_info*)C)->nmode, idx_C, ((struct tensor_info*)C)->strides, plan_ptr->P_nmode, plan_ptr->P_idx, &plan_ptr->P_strides_C);
    
    plan_ptr->I_size = calculate_size(plan_ptr->I_extents, plan_ptr->I_nmode);
    plan_ptr->B_size = calculate_size(plan_ptr->B_extents, plan_ptr->B_nmode);
    plan_ptr->P_size = calculate_size(plan_ptr->P_extents, plan_ptr->P_nmode);

    *plan = (TAPP_tensor_permute)plan_ptr;

    return 0;
}

int extract_P_indices(const int nmode_A, const int64_t* idx_A,
                      const int nmode_B, const int64_t* idx_B,
                      int64_t** P_idx_ptr)
{
    int max_P_nmode = nmode_A;
    if (nmode_B < max_P_nmode) max_P_nmode = nmode_B;
    *P_idx_ptr = malloc(max_P_nmode * sizeof(int64_t));
    int P_nmode = 0;
    for (size_t i = 0; i < nmode_A; i++)
    {
        bool in_B = false;
        for (size_t j = 0; j < nmode_B; j++)
        {
            if (idx_A[i] == idx_B[j]) {
                in_B = true;
                break;
            }
        }
        if (!in_B) continue;
        (*P_idx_ptr)[P_nmode] = idx_A[i];
        P_nmode++;
    }
    *P_idx_ptr = TAPP_realloc(*P_idx_ptr, P_nmode * sizeof(int64_t));
    return P_nmode;
}

int extract_IX_indices(const int nmode_X, const int64_t* idx_X,
                       const int nmode_Y, const int64_t* idx_y,
                       int64_t** IX_idx_ptr)
{
    int max_IX_nmode = nmode_X;
    *IX_idx_ptr = malloc(max_IX_nmode * sizeof(int64_t));
    int IX_nmode = 0;
    for (size_t i = 0; i < nmode_X; i++)
    {
        bool in_Y = false;
        for (size_t j = 0; j < nmode_Y; j++)
        {
            if (idx_X[i] == idx_y[j]) {
                in_Y = true;
                break;
            }
        }
        if (in_Y) continue;
        (*IX_idx_ptr)[IX_nmode] = idx_X[i];
        IX_nmode++;
    }
    *IX_idx_ptr = TAPP_realloc(*IX_idx_ptr, IX_nmode * sizeof(int64_t));
    return IX_nmode;
}

void extract_grouped_extents(const int nmode_X, const int64_t* idx_X, const int64_t* extents_X,
                           const int G_nmode, const int64_t* G_idx, int64_t** G_extents_X_ptr)
{
    *G_extents_X_ptr = malloc(G_nmode * sizeof(int64_t));
    for (size_t i = 0; i < G_nmode; i++)
    {
        (*G_extents_X_ptr)[i] = 0;
        for (size_t j = 0; j < nmode_X; j++)
        {
            if (G_idx[i] == idx_X[j]) {
                (*G_extents_X_ptr)[i] = extents_X[j];
                break;
            }
        }
    }
}

void extract_grouped_strides(const int nmode_X, const int64_t* idx_X, const int64_t* strides_X,
                           const int G_nmode, const int64_t* G_idx, int64_t** G_strides_X_ptr)
{
    *G_strides_X_ptr = malloc(G_nmode * sizeof(int64_t));
    for (size_t i = 0; i < G_nmode; i++)
    {
        (*G_strides_X_ptr)[i] = 0;
        for (size_t j = 0; j < nmode_X; j++)
        {
            if (G_idx[i] == idx_X[j]) {
                (*G_strides_X_ptr)[i] += strides_X[j];
                break;
            }
        }
    }
    
}

int64_t calculate_size(const int64_t* extents, const int nmode)
{
    int64_t size = 1;
    for (size_t i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    return size;
}

TAPP_error TAPP_destroy_tensor_permute(TAPP_tensor_permute plan)
{
    free(((struct plan*)plan)->I_idx);
    free(((struct plan*)plan)->B_idx);
    free(((struct plan*)plan)->P_idx);
    free(((struct plan*)plan)->I_extents);
    free(((struct plan*)plan)->B_extents);
    free(((struct plan*)plan)->P_extents);
    free(((struct plan*)plan)->I_strides);
    free(((struct plan*)plan)->B_strides);
    free(((struct plan*)plan)->P_strides_A);
    free(((struct plan*)plan)->P_strides_C);
    free((struct plan*)plan);
}
 
TAPP_error TAPP_execute_permute(TAPP_tensor_permute plan,
                                TAPP_executor exec,
                                TAPP_status* status,
                                const void* alpha,
                                const void* X,
                                const void* Y,
                                const void* beta,
                                    void* Z)
{
    struct plan* plan_ptr = (struct plan*)plan;

    int64_t* I_coords = malloc(plan_ptr->I_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->I_nmode; i++) I_coords[i] = 0;

    int64_t* P_coords = malloc(plan_ptr->P_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->P_nmode; i++) P_coords[i] = 0;

    int64_t* B_coords = malloc(plan_ptr->B_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->B_nmode; i++) B_coords[i] = 0;

    for (int64_t p = 0; p < plan_ptr->P_size; p++)
    {
        for (int64_t i = 0; i < plan_ptr->I_size; i++)
        {

            increment_coordinates(I_coords, plan_ptr->I_nmode, plan_ptr->I_extents);
        }

        for (int64_t b = 0; b < plan_ptr->B_size; b++)
        {
            increment_coordinates(B_coords, plan_ptr->B_nmode, plan_ptr->B_extents);
        }
        
        increment_coordinates(P_coords, plan_ptr->P_nmode, plan_ptr->P_extents);
    }
    
}

int calcualte_offset(int64_t* coords, int nmode, int64_t* strides)
{
    int index = 0;
    for (size_t i = 0; i < nmode; i++)
    {
        index += coords[i] * strides[i];
    }
    return index;
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