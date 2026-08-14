/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "../include/product.h"

int64_t calculate_size(const int64_t* extents, const int nmode);
int extract_H_indices(const int nmode_A, const int64_t* idx_A,
                      const int nmode_B, const int64_t* idx_B,
                      const int nmode_D, const int64_t* idx_D,
                      int64_t** H_idx_ptr);
int extract_P_indices(const int nmode_A, const int64_t* idx_A,
                      const int nmode_B, const int64_t* idx_B,
                      const int nmode_D, const int64_t* idx_D,
                      int64_t** P_idx_ptr);
int extract_FX_indices(const int nmode_X, const int64_t* idx_X,
                       const int nmode_Y, const int64_t* idx_y,
                       const int nmode_D, const int64_t* idx_D,
                       int64_t** FX_idx_ptr);
int extract_IX_indices(const int nmode_X, const int64_t* idx_X,
                       const int nmode_Y, const int64_t* idx_y,
                       const int nmode_Z, const int64_t* idx_Z,
                       int64_t** IX_idx_ptr);
void extract_grouped_extents(const int nmode_X, const int64_t* idx_X, const int64_t* extents_X,
                           const int G_nmode, const int64_t* G_idx, int64_t** G_extents_X_ptr);
void extract_grouped_strides(const int nmode_X, const int64_t* idx_X, const int64_t* strides_X,
                           const int G_nmode, const int64_t* G_idx, int64_t** G_strides_X_ptr);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
void sum_reduction(void* sum, const void* tensor, int index, TAPP_element_op op, TAPP_datatype type, TAPP_prectype prec);
void calculate_beta_C(const void* beta, const void* val_C, TAPP_datatype type_C, TAPP_element_op op_C, TAPP_prectype prec, void* accum, TAPP_datatype type_D);
void calculate_alpha_A_B(const void* alpha, const void* sum_A, TAPP_datatype type_A, const void* sum_B, TAPP_datatype type_B, TAPP_prectype prec, void* accum, TAPP_datatype type_D);
void calculate_op_D(void* accum, TAPP_datatype type_D, TAPP_element_op op_D, TAPP_prectype prec);
int calcualte_offset(int64_t* coords, int nmode, int64_t* strides);
void get_typed_value(void* val, const void* tensor, int64_t index, TAPP_datatype type, TAPP_prectype prec);
void assign_D(void* D, TAPP_datatype type_D, int64_t index_D, void* accum, TAPP_prectype prec);
int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int unique_idx_code);
int check_extents_pair(int nmode_X, const int64_t* idx_X, const int64_t* extents_X, int nmode_Y, const int64_t* idx_Y, const int64_t* extents_Y, int missmatch_code);
int check_same_structure(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_code, int idx_code, int extent_code);
int check_tensor_existence(const void* scalar, TAPP_datatype type, const void* tensor, int error_code);
int check_executor_existence(TAPP_executor exec, int error_code);
void* alloc_accum(TAPP_prectype prec, TAPP_datatype type);
void* alloc_typed_value(TAPP_prectype prec, TAPP_datatype type);
void* create_prec_scalar(const void* scalar, TAPP_datatype type, TAPP_prectype prec);
bool is_complex(TAPP_datatype type);
void set_typed_scalar_to_zero(void* sum, TAPP_prectype prec, TAPP_datatype type);
void set_typed_accum_to_zero(void* accum, TAPP_prectype prec, TAPP_datatype type);
bool is_equal(const void* val, TAPP_datatype type, const void* comp_val, TAPP_datatype comp_type);
void print_tensor_(int nmode, const int64_t* extents, const int64_t* strides, const void* data, TAPP_datatype type);

// calling realloc with size 0 is nonportable, this does the "right" thing
// see: https://valgrind.org/docs/manual/mc-manual.html#mc-manual.reallocsizezero
void* TAPP_realloc(void *ptr, size_t size) {
    if (size == 0) {
        if (ptr != NULL) free(ptr);
        return NULL;
    }
    else
        return realloc(ptr, size);
}

TAPP_error TAPP_create_tensor_product(TAPP_tensor_product* plan,
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
                                      TAPP_prectype prec)
{
    struct tensor_info* info_A_ptr = (struct tensor_info*)A;
    struct tensor_info* info_B_ptr = (struct tensor_info*)B;
    struct tensor_info* info_C_ptr = (struct tensor_info*)C;
    struct tensor_info* info_D_ptr = (struct tensor_info*)D;
    TAPP_error error_status = 0;
    if (error_status == 0) error_status = check_idx_occurrence(info_D_ptr->nmode, idx_D, info_A_ptr->nmode, idx_A, info_B_ptr->nmode, idx_B, 4);
    if (error_status == 0) error_status = check_extents_pair(info_A_ptr->nmode, idx_A, info_A_ptr->extents, info_A_ptr->nmode, idx_A, info_A_ptr->extents, 9);
    if (error_status == 0) error_status = check_extents_pair(info_B_ptr->nmode, idx_B, info_B_ptr->extents, info_B_ptr->nmode, idx_B, info_B_ptr->extents, 10);
    if (error_status == 0) error_status = check_extents_pair(info_D_ptr->nmode, idx_D, info_D_ptr->extents, info_D_ptr->nmode, idx_D, info_D_ptr->extents, 11);
    if (error_status == 0) error_status = check_extents_pair(info_A_ptr->nmode, idx_A, info_A_ptr->extents, info_B_ptr->nmode, idx_B, info_B_ptr->extents, 1);
    if (error_status == 0) error_status = check_extents_pair(info_A_ptr->nmode, idx_A, info_A_ptr->extents, info_D_ptr->nmode, idx_D, info_D_ptr->extents, 2);
    if (error_status == 0) error_status = check_extents_pair(info_B_ptr->nmode, idx_B, info_B_ptr->extents, info_D_ptr->nmode, idx_D, info_D_ptr->extents, 3);
    if (error_status == 0) error_status = check_same_structure(info_C_ptr->nmode, idx_C, info_C_ptr->extents, info_D_ptr->nmode ,idx_D ,info_D_ptr->extents ,5 ,6 ,7);
    if (error_status == 0) error_status = check_self_aliasing(info_D_ptr->nmode ,info_D_ptr->extents ,info_D_ptr->strides ,8);
    if (error_status != 0)
    {
        return error_status;
    }
    struct plan* plan_ptr = malloc(sizeof(struct plan));
    
    plan_ptr->A = A;

    plan_ptr->idx_A = malloc(((struct tensor_info*)A)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_A, idx_A, ((struct tensor_info*)A)->nmode * sizeof(int64_t));


    plan_ptr->B = B;

    plan_ptr->idx_B = malloc(((struct tensor_info*)B)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_B, idx_B, ((struct tensor_info*)B)->nmode * sizeof(int64_t));


    plan_ptr->C = C;

    plan_ptr->idx_C = malloc(((struct tensor_info*)C)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_C, idx_C, ((struct tensor_info*)C)->nmode * sizeof(int64_t));


    plan_ptr->D = D;

    plan_ptr->idx_D = malloc(((struct tensor_info*)D)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_D, idx_D, ((struct tensor_info*)D)->nmode * sizeof(int64_t));

    plan_ptr->type_A = info_A_ptr->type;
    plan_ptr->type_B = info_B_ptr->type;
    plan_ptr->type_C = info_C_ptr->type;
    plan_ptr->type_D = info_D_ptr->type;

    plan_ptr->op_A = op_A;
    plan_ptr->op_B = op_B;
    plan_ptr->op_C = op_C;
    plan_ptr->op_D = op_D;
    plan_ptr->prec = prec;

    plan_ptr->H_nmode = extract_H_indices(((struct tensor_info*)A)->nmode, idx_A,
                                          ((struct tensor_info*)B)->nmode, idx_B,
                                          ((struct tensor_info*)D)->nmode, idx_D,
                                          &plan_ptr->H_idx);
    plan_ptr->P_nmode = extract_P_indices(((struct tensor_info*)A)->nmode, idx_A,
                                          ((struct tensor_info*)B)->nmode, idx_B,
                                          ((struct tensor_info*)D)->nmode, idx_D,
                                          &plan_ptr->P_idx);
    plan_ptr->FA_nmode = extract_FX_indices(((struct tensor_info*)A)->nmode, idx_A,
                                            ((struct tensor_info*)B)->nmode, idx_B,
                                            ((struct tensor_info*)D)->nmode, idx_D,
                                            &plan_ptr->FA_idx);
    plan_ptr->FB_nmode = extract_FX_indices(((struct tensor_info*)B)->nmode, idx_B,
                                            ((struct tensor_info*)A)->nmode, idx_A,
                                            ((struct tensor_info*)D)->nmode, idx_D,
                                            &plan_ptr->FB_idx);
    plan_ptr->IA_nmode = extract_IX_indices(((struct tensor_info*)A)->nmode, idx_A,
                                            ((struct tensor_info*)B)->nmode, idx_B,
                                            ((struct tensor_info*)D)->nmode, idx_D,
                                            &plan_ptr->IA_idx);
    plan_ptr->IB_nmode = extract_IX_indices(((struct tensor_info*)B)->nmode, idx_B,
                                            ((struct tensor_info*)A)->nmode, idx_A,
                                            ((struct tensor_info*)D)->nmode, idx_D,
                                            &plan_ptr->IB_idx);

    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->H_nmode, plan_ptr->H_idx, &plan_ptr->H_extents);
    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->P_nmode, plan_ptr->P_idx, &plan_ptr->P_extents);
    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->FA_nmode, plan_ptr->FA_idx, &plan_ptr->FA_extents);
    extract_grouped_extents(((struct tensor_info*)B)->nmode, idx_B, ((struct tensor_info*)B)->extents, plan_ptr->FB_nmode, plan_ptr->FB_idx, &plan_ptr->FB_extents);
    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->IA_nmode, plan_ptr->IA_idx, &plan_ptr->IA_extents);
    extract_grouped_extents(((struct tensor_info*)B)->nmode, idx_B, ((struct tensor_info*)B)->extents, plan_ptr->IB_nmode, plan_ptr->IB_idx, &plan_ptr->IB_extents);
    
    extract_grouped_strides(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->strides, plan_ptr->H_nmode, plan_ptr->H_idx, &plan_ptr->H_strides_A);
    extract_grouped_strides(((struct tensor_info*)B)->nmode, idx_B, ((struct tensor_info*)B)->strides, plan_ptr->H_nmode, plan_ptr->H_idx, &plan_ptr->H_strides_B);
    extract_grouped_strides(((struct tensor_info*)D)->nmode, idx_D, ((struct tensor_info*)D)->strides, plan_ptr->H_nmode, plan_ptr->H_idx, &plan_ptr->H_strides_D);
    extract_grouped_strides(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->strides, plan_ptr->P_nmode, plan_ptr->P_idx, &plan_ptr->P_strides_A);
    extract_grouped_strides(((struct tensor_info*)B)->nmode, idx_B, ((struct tensor_info*)B)->strides, plan_ptr->P_nmode, plan_ptr->P_idx, &plan_ptr->P_strides_B);
    extract_grouped_strides(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->strides, plan_ptr->FA_nmode, plan_ptr->FA_idx, &plan_ptr->FA_strides_A);
    extract_grouped_strides(((struct tensor_info*)D)->nmode, idx_D, ((struct tensor_info*)D)->strides, plan_ptr->FA_nmode, plan_ptr->FA_idx, &plan_ptr->FA_strides_D);
    extract_grouped_strides(((struct tensor_info*)B)->nmode, idx_B, ((struct tensor_info*)B)->strides, plan_ptr->FB_nmode, plan_ptr->FB_idx, &plan_ptr->FB_strides_B);
    extract_grouped_strides(((struct tensor_info*)D)->nmode, idx_D, ((struct tensor_info*)D)->strides, plan_ptr->FB_nmode, plan_ptr->FB_idx, &plan_ptr->FB_strides_D);
    extract_grouped_strides(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->strides, plan_ptr->IA_nmode, plan_ptr->IA_idx, &plan_ptr->IA_strides_A);
    extract_grouped_strides(((struct tensor_info*)B)->nmode, idx_B, ((struct tensor_info*)B)->strides, plan_ptr->IB_nmode, plan_ptr->IB_idx, &plan_ptr->IB_strides_B);

    plan_ptr->H_size = calculate_size(plan_ptr->H_extents, plan_ptr->H_nmode);
    plan_ptr->P_size = calculate_size(plan_ptr->P_extents, plan_ptr->P_nmode);
    plan_ptr->FA_size = calculate_size(plan_ptr->FA_extents, plan_ptr->FA_nmode);
    plan_ptr->FB_size = calculate_size(plan_ptr->FB_extents, plan_ptr->FB_nmode);
    plan_ptr->IA_size = calculate_size(plan_ptr->IA_extents, plan_ptr->IA_nmode);
    plan_ptr->IB_size = calculate_size(plan_ptr->IB_extents, plan_ptr->IB_nmode);

    *plan = (TAPP_tensor_product)plan_ptr;

    return 0;
}

int extract_H_indices(const int nmode_A, const int64_t* idx_A,
                      const int nmode_B, const int64_t* idx_B,
                      const int nmode_D, const int64_t* idx_D,
                      int64_t** H_idx_ptr)
{
    int max_H_nmode = nmode_A;
    if (nmode_B < max_H_nmode) max_H_nmode = nmode_B;
    if (nmode_D < max_H_nmode) max_H_nmode = nmode_D;
    *H_idx_ptr = malloc(max_H_nmode * sizeof(int64_t));
    int H_nmode = 0;
    for (size_t i = 0; i < nmode_A; i++)
    {
        bool already_handled = false;
        for (size_t j = 0; j < i; j++)
        {
            if (idx_A[i] == idx_A[j]) {
                already_handled = true;
                break;
            }
        }
        if (already_handled) continue;

        bool in_B = false;
        for (size_t j = 0; j < nmode_B; j++)
        {
            if (idx_A[i] == idx_B[j]) {
                in_B = true;
                break;
            }
        }
        if (!in_B) continue;

        bool in_D = false;
        for (size_t j = 0; j < nmode_D; j++)
        {
            if (idx_A[i] == idx_D[j]) {
                in_D = true;
                break;
            }
        }
        if (!in_D) continue;

        (*H_idx_ptr)[H_nmode] = idx_A[i];
        H_nmode++;
    }
    *H_idx_ptr = TAPP_realloc(*H_idx_ptr, H_nmode * sizeof(int64_t));
    return H_nmode;
}

int extract_P_indices(const int nmode_A, const int64_t* idx_A,
                      const int nmode_B, const int64_t* idx_B,
                      const int nmode_D, const int64_t* idx_D,
                      int64_t** P_idx_ptr)
{
    int max_P_nmode = nmode_A;
    if (nmode_B < max_P_nmode) max_P_nmode = nmode_B;
    *P_idx_ptr = malloc(max_P_nmode * sizeof(int64_t));
    int P_nmode = 0;
    for (size_t i = 0; i < nmode_A; i++)
    {
        bool already_handled = false;
        for (size_t j = 0; j < i; j++)
        {
            if (idx_A[i] == idx_A[j]) {
                already_handled = true;
                break;
            }
        }
        if (already_handled) continue;

        bool in_B = false;
        for (size_t j = 0; j < nmode_B; j++)
        {
            if (idx_A[i] == idx_B[j]) {
                in_B = true;
                break;
            }
        }
        if (!in_B) continue;

        bool in_D = false;
        for (size_t j = 0; j < nmode_D; j++)
        {
            if (idx_A[i] == idx_D[j]) {
                in_D = true;
                break;
            }
        }
        if (in_D) continue;

        (*P_idx_ptr)[P_nmode] = idx_A[i];
        P_nmode++;
    }
    *P_idx_ptr = TAPP_realloc(*P_idx_ptr, P_nmode * sizeof(int64_t));
    return P_nmode;
}

int extract_FX_indices(const int nmode_X, const int64_t* idx_X,
                       const int nmode_Y, const int64_t* idx_y,
                       const int nmode_D, const int64_t* idx_D,
                       int64_t** FX_idx_ptr)
{
    int max_FX_nmode = nmode_X;
    *FX_idx_ptr = malloc(max_FX_nmode * sizeof(int64_t));
    int FX_nmode = 0;
    for (size_t i = 0; i < nmode_X; i++)
    {
        bool already_handled = false;
        for (size_t j = 0; j < i; j++)
        {
            if (idx_X[i] == idx_X[j]) {
                already_handled = true;
                break;
            }
        }
        if (already_handled) continue;

        bool in_Y = false;
        for (size_t j = 0; j < nmode_Y; j++)
        {
            if (idx_X[i] == idx_y[j]) {
                in_Y = true;
                break;
            }
        }
        if (in_Y) continue;

        bool in_D = false;
        for (size_t j = 0; j < nmode_D; j++)
        {
            if (idx_X[i] == idx_D[j]) {
                in_D = true;
                break;
            }
        }
        if (!in_D) continue;

        (*FX_idx_ptr)[FX_nmode] = idx_X[i];
        FX_nmode++;
    }
    *FX_idx_ptr = TAPP_realloc(*FX_idx_ptr, FX_nmode * sizeof(int64_t));
    return FX_nmode;
}

int extract_IX_indices(const int nmode_X, const int64_t* idx_X,
                       const int nmode_Y, const int64_t* idx_y,
                       const int nmode_Z, const int64_t* idx_Z,
                       int64_t** IX_idx_ptr)
{
    int max_IX_nmode = nmode_X;
    *IX_idx_ptr = malloc(max_IX_nmode * sizeof(int64_t));
    int IX_nmode = 0;
    for (size_t i = 0; i < nmode_X; i++)
    {
        bool already_handled = false;
        for (size_t j = 0; j < i; j++)
        {
            if (idx_X[i] == idx_X[j]) {
                already_handled = true;
                break;
            }
        }
        if (already_handled) continue;

        bool in_Y = false;
        for (size_t j = 0; j < nmode_Y; j++)
        {
            if (idx_X[i] == idx_y[j]) {
                in_Y = true;
                break;
            }
        }
        if (in_Y) continue;

        bool in_D = false;
        for (size_t j = 0; j < nmode_Z; j++)
        {
            if (idx_X[i] == idx_Z[j]) {
                in_D = true;
                break;
            }
        }
        if (in_D) continue;

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

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    free(((struct plan*)plan)->idx_A);
    free(((struct plan*)plan)->idx_B);
    free(((struct plan*)plan)->idx_C);
    free(((struct plan*)plan)->idx_D);
    free(((struct plan*)plan)->H_idx);
    free(((struct plan*)plan)->P_idx);
    free(((struct plan*)plan)->FA_idx);
    free(((struct plan*)plan)->FB_idx);
    free(((struct plan*)plan)->IA_idx);
    free(((struct plan*)plan)->IB_idx);
    free(((struct plan*)plan)->H_extents);
    free(((struct plan*)plan)->P_extents);
    free(((struct plan*)plan)->FA_extents);
    free(((struct plan*)plan)->FB_extents);
    free(((struct plan*)plan)->IA_extents);
    free(((struct plan*)plan)->IB_extents);
    free(((struct plan*)plan)->H_strides_A);
    free(((struct plan*)plan)->H_strides_B);
    free(((struct plan*)plan)->H_strides_D);
    free(((struct plan*)plan)->P_strides_A);
    free(((struct plan*)plan)->P_strides_B);
    free(((struct plan*)plan)->FA_strides_A);
    free(((struct plan*)plan)->FA_strides_D);
    free(((struct plan*)plan)->FB_strides_B);
    free(((struct plan*)plan)->FB_strides_D);
    free(((struct plan*)plan)->IA_strides_A);
    free(((struct plan*)plan)->IB_strides_B);
    free((struct plan*)plan);

    return 0;
}

TAPP_error TAPP_execute_product(TAPP_tensor_product plan,
                                TAPP_executor exec,
                                TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* B,
                                const void* beta,
                                const void* C,
                                void* D)
{
    struct plan* plan_ptr = (struct plan*)plan;

    TAPP_tensor_info info_A = (TAPP_tensor_info)(plan_ptr->A);
    struct tensor_info* info_A_ptr = (struct tensor_info*)(plan_ptr->A);
    
    TAPP_tensor_info info_B = (TAPP_tensor_info)(plan_ptr->B);
    struct tensor_info* info_B_ptr = (struct tensor_info*)(plan_ptr->B);

    TAPP_tensor_info info_C = (TAPP_tensor_info)(plan_ptr->C);
    struct tensor_info* info_C_ptr = (struct tensor_info*)(plan_ptr->C);

    TAPP_tensor_info info_D = (TAPP_tensor_info)(plan_ptr->D);
    struct tensor_info* info_D_ptr = (struct tensor_info*)(plan_ptr->D);

    int error_status = 0;

    if (error_status == 0) error_status = check_tensor_existence(beta, info_D_ptr->type, C, 12);
    if (error_status == 0) error_status = check_executor_existence(exec, 33);
    if (error_status != 0)
    {
        return error_status;
    }
    int64_t size_D;

    intptr_t* exec_ptr= &exec; //pointer to intptr_t (TAPP_executor)
    int* exec_int_ptr = (int*) *exec_ptr;//dereference to get the int pointer

    void* E_ = D;
    if((*exec_int_ptr) == 12 ) { // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
      size_D = calculate_size(info_D_ptr->extents, info_D_ptr->nmode);
      int64_t in_bytes;
      switch (info_D_ptr->type) { // tapp_datatype
      case TAPP_F32:
        in_bytes = (size_D)*(sizeof(float));
        break;
      case TAPP_F64:
        in_bytes = (size_D)*(sizeof(double));
        break;
      case TAPP_C32:
        in_bytes = (size_D)*(sizeof(float complex));
        break;
      case TAPP_C64:
        in_bytes = (size_D)*(sizeof(double complex));
        break;
      }
      E_ = malloc((size_t)in_bytes);
      memcpy(E_, D, (size_t)in_bytes);

    }

    if((*exec_int_ptr) == 2 || (*exec_int_ptr) == 12 ) { // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
        // if((*exec_int_ptr) == 2) printf("tapp used2 \n");

#ifdef TAPP_REFERENCE_ENABLE_TBLIS
        bind_tblis_execute_product(info_A_ptr->nmode, info_A_ptr->extents, info_A_ptr->strides, A, plan_ptr->op_A, plan_ptr->idx_A,
                                   info_B_ptr->nmode, info_B_ptr->extents, info_B_ptr->strides, B, plan_ptr->op_B, plan_ptr->idx_B,
                                   info_C_ptr->nmode, info_C_ptr->extents, info_C_ptr->strides, C, plan_ptr->op_C, plan_ptr->idx_C,
                                   info_D_ptr->nmode, info_D_ptr->extents, info_D_ptr->strides, E_, plan_ptr->op_D, plan_ptr->idx_D,
                                   alpha, beta, info_D_ptr->type);
#endif
    }

    if((*exec_int_ptr) == 1 || (*exec_int_ptr) == 12 ) { // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
        // if((*exec_int_ptr) == 1) printf("tapp used1 \n");

        void* accum = alloc_accum(plan_ptr->prec, plan_ptr->type_D);
        void* sum_A = alloc_typed_value(plan_ptr->prec, plan_ptr->type_A);
        void* sum_B = alloc_typed_value(plan_ptr->prec, plan_ptr->type_B);
        void* value_C = alloc_typed_value(plan_ptr->prec, plan_ptr->type_C);
        void* prec_alpha = create_prec_scalar(alpha, plan_ptr->type_D, plan_ptr->prec);
        void* prec_beta = create_prec_scalar(beta, plan_ptr->type_D, plan_ptr->prec);

        float value_zero = 0;
        bool beta_is_zero = !is_equal(beta, plan_ptr->type_D, &value_zero, TAPP_F32);
        
        int64_t* H_coords = malloc(plan_ptr->H_nmode * sizeof(int64_t));
        for (int i = 0; i < plan_ptr->H_nmode; i++) H_coords[i] = 0;

        int64_t* P_coords = malloc(plan_ptr->P_nmode * sizeof(int64_t));
        for (int i = 0; i < plan_ptr->P_nmode; i++) P_coords[i] = 0;

        int64_t* FA_coords = malloc(plan_ptr->FA_nmode * sizeof(int64_t));
        for (int i = 0; i < plan_ptr->FA_nmode; i++) FA_coords[i] = 0;

        int64_t* FB_coords = malloc(plan_ptr->FB_nmode * sizeof(int64_t));
        for (int i = 0; i < plan_ptr->FB_nmode; i++) FB_coords[i] = 0;

        int64_t* IA_coords = malloc(plan_ptr->IA_nmode * sizeof(int64_t));
        for (int i = 0; i < plan_ptr->IA_nmode; i++) IA_coords[i] = 0;

        int64_t* IB_coords = malloc(plan_ptr->IB_nmode * sizeof(int64_t));
        for (int i = 0; i < plan_ptr->IB_nmode; i++) IB_coords[i] = 0;

        for (int64_t h = 0; h < plan_ptr->H_size; h++)
        {
            int64_t H_offset_A = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_A);
            int64_t H_offset_B = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_B);
            int64_t H_offset_D = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_D);

            for (int64_t fa = 0; fa < plan_ptr->FA_size; fa++)
            {
                int64_t FA_offset_A = calcualte_offset(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_strides_A);
                int64_t FA_offset_D = calcualte_offset(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_strides_D);

                for (int64_t fb = 0; fb < plan_ptr->FB_size; fb++)
                {
                    int64_t FB_offset_B = calcualte_offset(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_strides_B);
                    int64_t FB_offset_D = calcualte_offset(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_strides_D);

                    int64_t offset_D = H_offset_D + FA_offset_D + FB_offset_D;

                    if (beta_is_zero)
                    {
                        get_typed_value(value_C, C, offset_D, plan_ptr->type_C, plan_ptr->prec);
                        calculate_beta_C(prec_beta, value_C, plan_ptr->type_C, plan_ptr->op_C, plan_ptr->prec, accum, plan_ptr->type_D);
                    }
                    else
                    {
                        set_typed_accum_to_zero(accum, plan_ptr->prec, plan_ptr->type_D);
                    }

                    for (int64_t p = 0; p < plan_ptr->P_size; p++)
                    {
                        int64_t P_offset_A = calcualte_offset(P_coords, plan_ptr->P_nmode, plan_ptr->P_strides_A);
                        int64_t P_offset_B = calcualte_offset(P_coords, plan_ptr->P_nmode, plan_ptr->P_strides_B);

                        set_typed_scalar_to_zero(sum_A, plan_ptr->prec, plan_ptr->type_A);
                        for (int64_t ia = 0; ia < plan_ptr->IA_size; ia++)
                        {
                            int64_t IA_offset_A = calcualte_offset(IA_coords, plan_ptr->IA_nmode, plan_ptr->IA_strides_A);
                            int64_t offset_A = H_offset_A + FA_offset_A + P_offset_A + IA_offset_A;
                            sum_reduction(sum_A, A, offset_A, plan_ptr->op_A, plan_ptr->type_A, plan_ptr->prec);
                            increment_coordinates(IA_coords, plan_ptr->IA_nmode, plan_ptr->IA_extents);
                        }

                        set_typed_scalar_to_zero(sum_B, plan_ptr->prec, plan_ptr->type_B);
                        for (int64_t ib = 0; ib < plan_ptr->IB_size; ib++)
                        {
                            int64_t IB_offset_B = calcualte_offset(IB_coords, plan_ptr->IB_nmode, plan_ptr->IB_strides_B);
                            int64_t offset_B = H_offset_B + FB_offset_B + P_offset_B + IB_offset_B;
                            sum_reduction(sum_B, B, offset_B, plan_ptr->op_B, plan_ptr->type_B, plan_ptr->prec);
                            increment_coordinates(IB_coords, plan_ptr->IB_nmode, plan_ptr->IB_extents);
                        }

                        calculate_alpha_A_B(prec_alpha, sum_A, plan_ptr->type_A, sum_B, plan_ptr->type_B, plan_ptr->prec, accum, plan_ptr->type_D);

                        increment_coordinates(P_coords, plan_ptr->P_nmode, plan_ptr->P_extents);
                    }

                    calculate_op_D(accum, plan_ptr->type_D, plan_ptr->op_D, plan_ptr->prec);

                    assign_D(D, plan_ptr->type_D, offset_D, accum, plan_ptr->prec);

                    increment_coordinates(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_extents);
                }

                increment_coordinates(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_extents);
            }
            
            increment_coordinates(H_coords, plan_ptr->H_nmode, plan_ptr->H_extents);
        }

        free(accum);
        free(sum_A);
        free(sum_B);
        free(value_C);
        free(prec_alpha);
        free(prec_beta);
        free(H_coords);
        free(P_coords);
        free(FA_coords);
        free(FB_coords);
        free(IA_coords);
        free(IB_coords);
    }

    bool comp_ = true;
    if((*exec_int_ptr) == 12 ) { // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
#ifdef TAPP_REFERENCE_ENABLE_TBLIS
      comp_ = compare_tensors_(D, E_, (int64_t)size_D, plan_ptr->type_D);
#endif
      if(!comp_){
        printf("A: \n");
        print_tensor_(info_A_ptr->nmode, info_A_ptr->extents, info_A_ptr->strides, A, info_A_ptr->type);
        printf("B: \n");
        print_tensor_(info_B_ptr->nmode, info_B_ptr->extents, info_B_ptr->strides, B, info_B_ptr->type);
        printf("C: \n");
        print_tensor_(info_C_ptr->nmode, info_C_ptr->extents, info_C_ptr->strides, C, info_C_ptr->type);
        printf("D: \n");
        print_tensor_(info_D_ptr->nmode, info_D_ptr->extents, info_D_ptr->strides, D, info_D_ptr->type);
        printf("E_: \n");
        print_tensor_(info_D_ptr->nmode, info_D_ptr->extents, info_D_ptr->strides, E_, info_D_ptr->type);
        printf("alpha: \n");
        print_tensor_(0, info_D_ptr->extents, info_D_ptr->strides, alpha, info_D_ptr->type);
        printf("beta: \n");
        print_tensor_(0, info_D_ptr->extents, info_D_ptr->strides, beta, info_D_ptr->type);
        printf("size_D: %d \n", (int)size_D);
        printf("nmode_D: %d \n", info_D_ptr->nmode);
      }
      free(E_);
    }

    if(!comp_) return 137;
    return 0;
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

void print_tensor_(int nmode, const int64_t* extents, const int64_t* strides, const void* data_, TAPP_datatype type) {

    int64_t* coords;
    if(nmode > 0) coords = malloc(nmode * sizeof(int64_t));
    else {
      printf("scalar");
    }
    int64_t size = 1;
    for (size_t i = 0; i < nmode; i++)
    {
        coords[i] = 0;
        size *= extents[i];
    }
    printf("\t");
    for (size_t i = 0; i < size; i++)
    {
        int64_t index = 0;
        for (size_t i = 0; i < nmode; i++)
        {
            index += coords[i] * strides[i];
        }
        switch (type) { // tapp_datatype
          case TAPP_F32:
          {
            float* datas = (float*) data_;
            printf("%.3f", datas[index]);
            break;
          }
          case TAPP_F64:
          {
            double* datad = (double*) data_;
            printf("%.3f", datad[index]);
            break;
          }
          case TAPP_C32:
          {
            float complex* datac = (float complex*) data_;
            printf("%.3f+%.3fi", crealf(datac[index]), cimagf(datac[index]));
            break;
          }
          case TAPP_C64:
          {
            double complex* dataz = (double complex*) data_;
            printf("%.3f+%.3fi", creal(dataz[index]), cimag(dataz[index]));
            break;
          }
        }

        if (nmode <= 0) continue;

        int k = 0;
        do
        {
            if (k != 0) {
                printf("\n");
                if (i < size - 1) {
                    printf("\t");
                }
            }
            else {
                printf(" ");
            }
            coords[k] = (coords[k] + 1) % extents[k];
            k++;
        } while (coords[k - 1] == 0 && k < nmode);
    }

    if(nmode > 0) free(coords);
    else printf("\n");
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

int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int unique_idx_code)
{
    for (size_t i = 0; i < nmode_origin; i++)
    {
        int idx_found = 0;
        for (size_t j = 0; j < nmode_test_A; j++)
        {
            if (idx_origin[i] == idx_test_A[j])
            {
                idx_found++;
                break;
            }
        }
        for (size_t j = 0; j < nmode_test_B; j++)
        {
            if (idx_origin[i] == idx_test_B[j])
            {
                idx_found++;
                break;
            }
        }
        if (idx_found == 0)
        { //No other occurrence, error
            return unique_idx_code;
        }
    }
    return 0;
}

int check_extents_pair(int nmode_X, const int64_t* idx_X, const int64_t* extents_X, int nmode_Y, const int64_t* idx_Y, const int64_t* extents_Y, int missmatch_code)
{
    for (size_t i = 0; i < nmode_X; i++)
    {
        for (size_t j = 0; j < nmode_Y; j++)
        {
            if (idx_X[i] == idx_Y[j] && extents_X[i] != extents_Y[j])
            {
                return missmatch_code;
            }
        }
    }
    return 0;
}

int check_same_structure(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_code, int idx_code, int extent_code)
{
    if(nmode_A != nmode_B)
    {
        return nmode_code;
    }

    for (size_t i = 0; i < nmode_B; i++)
    {
        if (idx_B[i] != idx_A[i])
        {
            return idx_code;
        }
        if (extents_B[i] != extents_A[i])
        {
            return extent_code;
        }
    }
    return 0;
}

int check_tensor_existence(const void* scalar, TAPP_datatype type, const void* tensor, int error_code)
{
    float value_zero = 0;
    return tensor == NULL && !is_equal(scalar, type, &value_zero, TAPP_F32) ? error_code : 0;
}

int check_executor_existence(TAPP_executor exec, int error_code)
{
    if(!exec) return error_code;
    intptr_t* exec_ptr= &exec; //pointer to intptr_t (TAPP_executor)
    int* eip = (int*) *exec_ptr;//dereference to get the int pointer
    if((*eip) == 1 || (*eip) == 2 ||  (*eip) == 12) return 0;
    return error_code; // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
}

/* TAPP_ID is a identity macro for datatypes that can not be conjugated*/
#define TAPP_ID(x) (x)

/* TAPP_DATATYPE_LIST is an X-macro: nesting one use of it inside another
 * (e.g. a switch(type_A) inside a switch(type_accum), both driven by this
 * same list) hits the C preprocessor's self-recursion guard - a macro
 * cannot re-expand itself while its own expansion is still being
 * rescanned, even indirectly through several layers of callback. That
 * rules out any shared helper macro between nesting levels too (it hits
 * the same guard one level down), so TAPP_DATATYPE_LIST2/3/4 below are
 * fully independent, identically-defined copies - use the Nth copy for
 * the Nth level of nesting. */
#if defined(TAPP_REFERENCE_ENABLE_F16) && defined(TAPP_REFERENCE_ENABLE_BF16)
#define TAPP_DATATYPE_LIST(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_F16, _Float16, TAPP_ID) \
    X(TAPP_BF16, __bf16, TAPP_ID)
#define TAPP_DATATYPE_LIST2(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_F16, _Float16, TAPP_ID) \
    X(TAPP_BF16, __bf16, TAPP_ID)
#define TAPP_DATATYPE_LIST3(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_F16, _Float16, TAPP_ID) \
    X(TAPP_BF16, __bf16, TAPP_ID)
#define TAPP_DATATYPE_LIST4(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_F16, _Float16, TAPP_ID) \
    X(TAPP_BF16, __bf16, TAPP_ID)
#define TAPP_REAL_TYPES(X) \
    X(TAPP_F32, float) \
    X(TAPP_F64, double) \
    X(TAPP_F16, _Float16) \
    X(TAPP_BF16, __bf16)
#elif defined(TAPP_REFERENCE_ENABLE_F16)
#define TAPP_DATATYPE_LIST(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_F16, _Float16, TAPP_ID)
#define TAPP_DATATYPE_LIST2(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_F16, _Float16, TAPP_ID)
#define TAPP_DATATYPE_LIST3(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_F16, _Float16, TAPP_ID)
#define TAPP_DATATYPE_LIST4(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_F16, _Float16, TAPP_ID)
#define TAPP_REAL_TYPES(X) \
    X(TAPP_F32, float) \
    X(TAPP_F64, double) \
    X(TAPP_F16, _Float16)
#elif defined(TAPP_REFERENCE_ENABLE_BF16)
#define TAPP_DATATYPE_LIST(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_BF16, __bf16, TAPP_ID)
#define TAPP_DATATYPE_LIST2(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_BF16, __bf16, TAPP_ID)
#define TAPP_DATATYPE_LIST3(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_BF16, __bf16, TAPP_ID)
#define TAPP_DATATYPE_LIST4(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj) \
    X(TAPP_BF16, __bf16, TAPP_ID)
#define TAPP_REAL_TYPES(X) \
    X(TAPP_F32, float) \
    X(TAPP_F64, double) \
    X(TAPP_BF16, __bf16)
#else
#define TAPP_DATATYPE_LIST(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj)
#define TAPP_DATATYPE_LIST2(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj)
#define TAPP_DATATYPE_LIST3(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj)
#define TAPP_DATATYPE_LIST4(X) \
    X(TAPP_F32, float, TAPP_ID) \
    X(TAPP_F64, double, TAPP_ID) \
    X(TAPP_C32, complex float, conjf) \
    X(TAPP_C64, complex double, conj)
#define TAPP_REAL_TYPES(X) \
    X(TAPP_F32, float) \
    X(TAPP_F64, double)
#endif

/* X(ENUM, CTYPE, CONJFN) for the complex-valued TAPP_datatype values */
#define TAPP_COMPLEX_TYPES(X)   \
    X(TAPP_C32, complex float, conjf)  \
    X(TAPP_C64, complex double, conj)

#define TAPP_ALLOC_CASE(ENUM, T, CONJFN) case ENUM: return malloc(sizeof(T));

void* alloc_accum(TAPP_prectype prec, TAPP_datatype type)
{
    bool is_complex_type = is_complex(type);
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type) { TAPP_DATATYPE_LIST(TAPP_ALLOC_CASE) default: return NULL; }
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        return malloc(is_complex_type ? sizeof(complex float) : sizeof(float));
    case TAPP_F64F64_ACCUM_F64:
        return malloc(is_complex_type ? sizeof(complex double) : sizeof(double));
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        return malloc(sizeof(_Float16));
#endif
    default:
        return NULL;
    }
}

void* alloc_typed_value(TAPP_prectype prec, TAPP_datatype type)
{
    bool is_complex_type = is_complex(type);
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type) { TAPP_DATATYPE_LIST(TAPP_ALLOC_CASE) default: return NULL; }
        break;
    case TAPP_F32F32_ACCUM_F32:
        return malloc(is_complex_type ? sizeof(complex float) : sizeof(float));
    case TAPP_F64F64_ACCUM_F64:
        return malloc(is_complex_type ? sizeof(complex double) : sizeof(double));
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
    case TAPP_F16F16_ACCUM_F32:
        return malloc(is_complex_type ? sizeof(complex _Float16) : sizeof(_Float16));
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        return malloc(is_complex_type ? sizeof(complex __bf16) : sizeof(__bf16));
#endif
    default:
        return NULL;
    }
}
#undef TAPP_ALLOC_CASE

/* The computational precision to use when the caller passed TAPP_DEFAULT_PREC,
 * i.e. the "native" precision of a storage type. */
static TAPP_prectype default_prec_for_type(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
    case TAPP_C32:
        return TAPP_F32F32_ACCUM_F32;
    case TAPP_F64:
    case TAPP_C64:
        return TAPP_F64F64_ACCUM_F64;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16:
        return TAPP_F16F16_ACCUM_F16;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16:
        return TAPP_BF16BF16_ACCUM_F32;
#endif
    default:
        return TAPP_F32F32_ACCUM_F32;
    }
}

/* The do while is to give each case its own scope and unlike {} it won't break if/else pairing */
#define TAPP_MAKE_PREC_SCALAR(T_SRC, T_TARGET) \
    do { T_TARGET* p = malloc(sizeof(T_TARGET)); *p = *(T_SRC*)scalar; return p; } while (0)

#ifdef TAPP_REFERENCE_ENABLE_F16
#define TAPP_PREC_SCALAR_F16_REAL(T_SRC) \
    case TAPP_F16F16_ACCUM_F16: case TAPP_F16F16_ACCUM_F32: TAPP_MAKE_PREC_SCALAR(T_SRC, _Float16);
#define TAPP_PREC_SCALAR_F16_CPLX(T_SRC) \
    case TAPP_F16F16_ACCUM_F16: case TAPP_F16F16_ACCUM_F32: TAPP_MAKE_PREC_SCALAR(T_SRC, complex _Float16);
#else
#define TAPP_PREC_SCALAR_F16_REAL(T_SRC)
#define TAPP_PREC_SCALAR_F16_CPLX(T_SRC)
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
#define TAPP_PREC_SCALAR_BF16_REAL(T_SRC) \
    case TAPP_BF16BF16_ACCUM_F32: TAPP_MAKE_PREC_SCALAR(T_SRC, __bf16);
#define TAPP_PREC_SCALAR_BF16_CPLX(T_SRC) \
    case TAPP_BF16BF16_ACCUM_F32: TAPP_MAKE_PREC_SCALAR(T_SRC, complex __bf16);
#else
#define TAPP_PREC_SCALAR_BF16_REAL(T_SRC)
#define TAPP_PREC_SCALAR_BF16_CPLX(T_SRC)
#endif

#define TAPP_PREC_SCALAR_CASE_REAL(ENUM, T_SRC)     \
    case ENUM:                                       \
        switch (prec)                                \
        {                                             \
        case TAPP_F32F32_ACCUM_F32: TAPP_MAKE_PREC_SCALAR(T_SRC, float);   \
        case TAPP_F64F64_ACCUM_F64: TAPP_MAKE_PREC_SCALAR(T_SRC, double);  \
        TAPP_PREC_SCALAR_F16_REAL(T_SRC)             \
        TAPP_PREC_SCALAR_BF16_REAL(T_SRC)            \
        default: return NULL;                        \
        }

#define TAPP_PREC_SCALAR_CASE_CPLX(ENUM, T_SRC, CONJFN)  \
    case ENUM:                                       \
        switch (prec)                                \
        {                                             \
        case TAPP_F32F32_ACCUM_F32: TAPP_MAKE_PREC_SCALAR(T_SRC, complex float);   \
        case TAPP_F64F64_ACCUM_F64: TAPP_MAKE_PREC_SCALAR(T_SRC, complex double);  \
        TAPP_PREC_SCALAR_F16_CPLX(T_SRC)             \
        TAPP_PREC_SCALAR_BF16_CPLX(T_SRC)            \
        default: return NULL;                        \
        }

void* create_prec_scalar(const void* scalar, TAPP_datatype type, TAPP_prectype prec)
{
    if (prec == TAPP_DEFAULT_PREC)
        prec = default_prec_for_type(type);
    switch (type)
    {
    TAPP_REAL_TYPES(TAPP_PREC_SCALAR_CASE_REAL)
    TAPP_COMPLEX_TYPES(TAPP_PREC_SCALAR_CASE_CPLX)
    default:
        return NULL;
    }
}
#undef TAPP_PREC_SCALAR_CASE_CPLX
#undef TAPP_PREC_SCALAR_CASE_REAL
#undef TAPP_PREC_SCALAR_BF16_CPLX
#undef TAPP_PREC_SCALAR_BF16_REAL
#undef TAPP_PREC_SCALAR_F16_CPLX
#undef TAPP_PREC_SCALAR_F16_REAL
#undef TAPP_MAKE_PREC_SCALAR

bool is_complex(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_C32:
    case TAPP_C64:
        return true;
    default:
        return false;
    }
}

#define TAPP_ZERO_CASE(ENUM, T, CONJFN) case ENUM: *(T*)ptr = 0; break;

void set_typed_scalar_to_zero(void* sum, TAPP_prectype prec, TAPP_datatype type)
{
    void* ptr = sum;
    bool is_complex_type = is_complex(type);
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type) { TAPP_DATATYPE_LIST(TAPP_ZERO_CASE) default: break; }
        break;
    case TAPP_F32F32_ACCUM_F32:
        if (is_complex_type) *(complex float*)ptr = 0; else *(float*)ptr = 0;
        break;
    case TAPP_F64F64_ACCUM_F64:
        if (is_complex_type) *(complex double*)ptr = 0; else *(double*)ptr = 0;
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
    case TAPP_F16F16_ACCUM_F32:
        if (is_complex_type) *(complex _Float16*)ptr = 0; else *(_Float16*)ptr = 0;
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        if (is_complex_type) *(complex __bf16*)ptr = 0; else *(__bf16*)ptr = 0;
        break;
#endif
    default:
        break;
    }
}

void set_typed_accum_to_zero(void* accum, TAPP_prectype prec, TAPP_datatype type)
{
    void* ptr = accum;
    bool is_complex_type = is_complex(type);
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type) { TAPP_DATATYPE_LIST(TAPP_ZERO_CASE) default: break; }
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        if (is_complex_type) *(complex float*)ptr = 0; else *(float*)ptr = 0;
        break;
    case TAPP_F64F64_ACCUM_F64:
        if (is_complex_type) *(complex double*)ptr = 0; else *(double*)ptr = 0;
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        if (is_complex_type) *(complex _Float16*)ptr = 0; else *(_Float16*)ptr = 0;
        break;
#endif
    default:
        break;
    }
}
#undef TAPP_ZERO_CASE

#define TAPP_IS_EQUAL_INNER(ENUM, T_COMP, CONJFN) \
    case ENUM: return *(T_VAL*)val == *(T_COMP*)comp_val;
#define TAPP_IS_EQUAL_OUTER(ENUM, T, CONJFN)                                             \
    case ENUM: {                                                                          \
        typedef T T_VAL;                                                                  \
        switch (comp_type) { TAPP_DATATYPE_LIST2(TAPP_IS_EQUAL_INNER) default: return false; } \
    }

bool is_equal(const void* val, TAPP_datatype type, const void* comp_val, TAPP_datatype comp_type)
{
    switch (type) { TAPP_DATATYPE_LIST(TAPP_IS_EQUAL_OUTER) default: return false; }
}
#undef TAPP_IS_EQUAL_OUTER
#undef TAPP_IS_EQUAL_INNER

#define TAPP_SUM_SELF_CASE(ENUM, T, CONJFN) \
    case ENUM: *(T*)sum += (op == TAPP_CONJUGATE) ? CONJFN(((T*)tensor)[index]) : ((T*)tensor)[index]; break;
#define TAPP_SUM_REAL_CASE(ENUM, T) \
    case ENUM: *(T_SUM_R*)sum += ((T*)tensor)[index]; break;
#define TAPP_SUM_CPLX_CASE(ENUM, T, CONJFN) \
    case ENUM: *(T_SUM_C*)sum += (op == TAPP_CONJUGATE) ? CONJFN(((T*)tensor)[index]) : ((T*)tensor)[index]; break;

void sum_reduction(void* sum, const void* tensor, int index, TAPP_element_op op, TAPP_datatype type, TAPP_prectype prec)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type) { TAPP_DATATYPE_LIST(TAPP_SUM_SELF_CASE) default: break; }
        break;
    case TAPP_F32F32_ACCUM_F32:
        {
            typedef float T_SUM_R; typedef complex float T_SUM_C;
            switch (type) { TAPP_REAL_TYPES(TAPP_SUM_REAL_CASE) TAPP_COMPLEX_TYPES(TAPP_SUM_CPLX_CASE) default: break; }
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        {
            typedef double T_SUM_R; typedef complex double T_SUM_C;
            switch (type) { TAPP_REAL_TYPES(TAPP_SUM_REAL_CASE) TAPP_COMPLEX_TYPES(TAPP_SUM_CPLX_CASE) default: break; }
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
    case TAPP_F16F16_ACCUM_F32:
        {
            typedef _Float16 T_SUM_R; typedef complex _Float16 T_SUM_C;
            switch (type) { TAPP_REAL_TYPES(TAPP_SUM_REAL_CASE) TAPP_COMPLEX_TYPES(TAPP_SUM_CPLX_CASE) default: break; }
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        {
            typedef __bf16 T_SUM_R; typedef complex __bf16 T_SUM_C;
            switch (type) { TAPP_REAL_TYPES(TAPP_SUM_REAL_CASE) TAPP_COMPLEX_TYPES(TAPP_SUM_CPLX_CASE) default: break; }
        }
        break;
#endif
    default:
        break;
    }
}
#undef TAPP_SUM_CPLX_CASE
#undef TAPP_SUM_REAL_CASE
#undef TAPP_SUM_SELF_CASE

#define TAPP_BETA_CASE(BETA_ENUM, T_BETA, BETA_CONJFN) \
    case BETA_ENUM: \
        *(T_ACCUM*)accum = *(T_BETA*)beta * val_C_v; \
        break;
#define TAPP_C_CASE(C_ENUM, T_C_TY, C_CONJFN)                                                    \
    case C_ENUM: {                                                                                \
        typedef T_C_TY T_C;                                                                       \
        T_C val_C_v = (op_C == TAPP_CONJUGATE) ? C_CONJFN(*(T_C*)val_C) : *(T_C*)val_C;           \
        switch (type_D) { TAPP_DATATYPE_LIST3(TAPP_BETA_CASE) default: break; }                    \
        break;                                                                                    \
    }
#define TAPP_ACCUM_CASE_BETA(ACCUM_ENUM, T_ACCUM_TY, ACCUM_CONJFN)  \
    case ACCUM_ENUM: {                                              \
        typedef T_ACCUM_TY T_ACCUM;                                 \
        switch (type_C) { TAPP_DATATYPE_LIST2(TAPP_C_CASE) default: break; } \
        break;                                                      \
    }
/* beta is always type_D, same as accum, so beta's complexness always matches
 * accum's - only is_complex_C varies independently. (The do-while is to give
 * each case its own scope; unlike {} it won't break if/else pairing.) */
#define TAPP_CALC_BETA_C_PREC_CASE(T_IN_R, T_IN_C, T_ACC_R, T_ACC_C)               \
    do {                                                                            \
        if (is_complex_D) {                                                         \
            if (is_complex_C) *(T_ACC_C*)accum = *(T_IN_C*)beta * *(T_IN_C*)val_C;  \
            else               *(T_ACC_C*)accum = *(T_IN_C*)beta * *(T_IN_R*)val_C; \
        } else {                                                                    \
            if (is_complex_C) *(T_ACC_R*)accum = *(T_IN_R*)beta * *(T_IN_C*)val_C;  \
            else               *(T_ACC_R*)accum = *(T_IN_R*)beta * *(T_IN_R*)val_C; \
        }                                                                           \
    } while (0)

void calculate_beta_C(const void* beta, const void* val_C, TAPP_datatype type_C, TAPP_element_op op_C, TAPP_prectype prec, void* accum, TAPP_datatype type_D)
{
    bool is_complex_D = is_complex(type_D);
    bool is_complex_C = is_complex(type_C);
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type_D) { TAPP_DATATYPE_LIST(TAPP_ACCUM_CASE_BETA) default: break; }
        break;
    case TAPP_F32F32_ACCUM_F32:
        TAPP_CALC_BETA_C_PREC_CASE(float, complex float, float, complex float);
        break;
    case TAPP_F64F64_ACCUM_F64:
        TAPP_CALC_BETA_C_PREC_CASE(double, complex double, double, complex double);
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        TAPP_CALC_BETA_C_PREC_CASE(_Float16, complex _Float16, _Float16, complex _Float16);
        break;
    case TAPP_F16F16_ACCUM_F32:
        TAPP_CALC_BETA_C_PREC_CASE(_Float16, complex _Float16, float, complex float);
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        TAPP_CALC_BETA_C_PREC_CASE(__bf16, complex __bf16, float, complex float);
        break;
#endif
    default:
        break;
    }
}
#undef TAPP_CALC_BETA_C_PREC_CASE
#undef TAPP_ACCUM_CASE_BETA
#undef TAPP_C_CASE
#undef TAPP_BETA_CASE

#define TAPP_ALPHA_CASE(ALPHA_ENUM, T_ALPHA, ALPHA_CONJFN) \
    case ALPHA_ENUM: \
        *(T_ACCUM*)accum += *(T_ALPHA*)alpha * *(T_A*)sum_A * *(T_B*)sum_B; \
        break;
#define TAPP_B_CASE(B_ENUM, T_B_TY, B_CONJFN)                                        \
    case B_ENUM: {                                                                    \
        typedef T_B_TY T_B;                                                           \
        switch (type_D) { TAPP_DATATYPE_LIST4(TAPP_ALPHA_CASE) default: break; }       \
        break;                                                                        \
    }
#define TAPP_A_CASE(A_ENUM, T_A_TY, A_CONJFN)                                \
    case A_ENUM: {                                                            \
        typedef T_A_TY T_A;                                                   \
        switch (type_B) { TAPP_DATATYPE_LIST3(TAPP_B_CASE) default: break; }   \
        break;                                                                \
    }
#define TAPP_ACCUM_CASE_AB(ACCUM_ENUM, T_ACCUM_TY, ACCUM_CONJFN)             \
    case ACCUM_ENUM: {                                                        \
        typedef T_ACCUM_TY T_ACCUM;                                           \
        switch (type_A) { TAPP_DATATYPE_LIST2(TAPP_A_CASE) default: break; }   \
        break;                                                                \
    }
/* alpha is always type_D, same as accum, so alpha's complexness always
 * matches accum's - only is_complex_A/is_complex_B vary independently. */
#define TAPP_CALC_ALPHA_A_B_PREC_CASE(T_IN_R, T_IN_C, T_ACC_R, T_ACC_C)                                          \
    do {                                                                                                          \
        if (is_complex_D) {                                                                                      \
            if (is_complex_A) {                                                                                  \
                if (is_complex_B) *(T_ACC_C*)accum += *(T_IN_C*)alpha * *(T_IN_C*)sum_A * *(T_IN_C*)sum_B;        \
                else              *(T_ACC_C*)accum += *(T_IN_C*)alpha * *(T_IN_C*)sum_A * *(T_IN_R*)sum_B;        \
            } else {                                                                                              \
                if (is_complex_B) *(T_ACC_C*)accum += *(T_IN_C*)alpha * *(T_IN_R*)sum_A * *(T_IN_C*)sum_B;        \
                else              *(T_ACC_C*)accum += *(T_IN_C*)alpha * *(T_IN_R*)sum_A * *(T_IN_R*)sum_B;        \
            }                                                                                                     \
        } else {                                                                                                  \
            if (is_complex_A) {                                                                                  \
                if (is_complex_B) *(T_ACC_R*)accum += *(T_IN_R*)alpha * *(T_IN_C*)sum_A * *(T_IN_C*)sum_B;        \
                else              *(T_ACC_R*)accum += *(T_IN_R*)alpha * *(T_IN_C*)sum_A * *(T_IN_R*)sum_B;        \
            } else {                                                                                              \
                if (is_complex_B) *(T_ACC_R*)accum += *(T_IN_R*)alpha * *(T_IN_R*)sum_A * *(T_IN_C*)sum_B;        \
                else              *(T_ACC_R*)accum += *(T_IN_R*)alpha * *(T_IN_R*)sum_A * *(T_IN_R*)sum_B;        \
            }                                                                                                     \
        }                                                                                                         \
    } while (0)

void calculate_alpha_A_B(const void* alpha, const void* sum_A, TAPP_datatype type_A, const void* sum_B, TAPP_datatype type_B, TAPP_prectype prec, void* accum, TAPP_datatype type_D)
{
    bool is_complex_D = is_complex(type_D);
    bool is_complex_A = is_complex(type_A);
    bool is_complex_B = is_complex(type_B);
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type_D) { TAPP_DATATYPE_LIST(TAPP_ACCUM_CASE_AB) default: break; }
        break;
    case TAPP_F32F32_ACCUM_F32:
        TAPP_CALC_ALPHA_A_B_PREC_CASE(float, complex float, float, complex float);
        break;
    case TAPP_F64F64_ACCUM_F64:
        TAPP_CALC_ALPHA_A_B_PREC_CASE(double, complex double, double, complex double);
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        TAPP_CALC_ALPHA_A_B_PREC_CASE(_Float16, complex _Float16, _Float16, complex _Float16);
        break;
    case TAPP_F16F16_ACCUM_F32:
        TAPP_CALC_ALPHA_A_B_PREC_CASE(_Float16, complex _Float16, float, complex float);
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        TAPP_CALC_ALPHA_A_B_PREC_CASE(__bf16, complex __bf16, float, complex float);
        break;
#endif
    default:
        break;
    }
}
#undef TAPP_CALC_ALPHA_A_B_PREC_CASE
#undef TAPP_ACCUM_CASE_AB
#undef TAPP_A_CASE
#undef TAPP_B_CASE
#undef TAPP_ALPHA_CASE

void calculate_op_D(void* accum, TAPP_datatype type_D, TAPP_element_op op_D, TAPP_prectype prec)
{
    if (op_D != TAPP_CONJUGATE || !is_complex(type_D))
        return;
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        if (type_D == TAPP_C32) *(complex float*)accum = conjf(*(complex float*)accum);
        else                    *(complex double*)accum = conj(*(complex double*)accum);
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        *(complex float*)accum = conjf(*(complex float*)accum);
        break;
    case TAPP_F64F64_ACCUM_F64:
        *(complex double*)accum = conj(*(complex double*)accum);
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        break;
#endif
    default:
        break;
    }
}

#define TAPP_GET_VAL_ID_CASE(ENUM, T, CONJFN) \
    case ENUM: *(T*)val = ((T*)tensor)[index]; break;
#define TAPP_GET_VAL_REAL_CASE(ENUM, T) \
    case ENUM: *(T_TARGET_R*)val = ((T*)tensor)[index]; break;
#define TAPP_GET_VAL_CPLX_CASE(ENUM, T, CONJFN) \
    case ENUM: *(T_TARGET_C*)val = ((T*)tensor)[index]; break;
#define TAPP_GET_VAL_PREC_CASE(T_R, T_C)                                    \
    {                                                                        \
        typedef T_R T_TARGET_R; typedef T_C T_TARGET_C;                     \
        switch (type)                                                        \
        {                                                                    \
        TAPP_REAL_TYPES(TAPP_GET_VAL_REAL_CASE)                              \
        TAPP_COMPLEX_TYPES(TAPP_GET_VAL_CPLX_CASE)                           \
        default: break;                                                      \
        }                                                                    \
    }

void get_typed_value(void* val, const void* tensor, int64_t index, TAPP_datatype type, TAPP_prectype prec)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type) { TAPP_DATATYPE_LIST(TAPP_GET_VAL_ID_CASE) default: break; }
        break;
    case TAPP_F32F32_ACCUM_F32:
        TAPP_GET_VAL_PREC_CASE(float, complex float);
        break;
    case TAPP_F64F64_ACCUM_F64:
        TAPP_GET_VAL_PREC_CASE(double, complex double);
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
    case TAPP_F16F16_ACCUM_F32:
        TAPP_GET_VAL_PREC_CASE(_Float16, complex _Float16);
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        TAPP_GET_VAL_PREC_CASE(__bf16, complex __bf16);
        break;
#endif
    default:
        break;
    }
}
#undef TAPP_GET_VAL_PREC_CASE
#undef TAPP_GET_VAL_CPLX_CASE
#undef TAPP_GET_VAL_REAL_CASE
#undef TAPP_GET_VAL_ID_CASE

#define TAPP_ASSIGN_ID_CASE(ENUM, T, CONJFN) \
    case ENUM: ((T*)D)[index_D] = *(T*)accum; break;
#define TAPP_ASSIGN_REAL_CASE(ENUM, T) \
    case ENUM: ((T*)D)[index_D] = *(T_ACC_R*)accum; break;
#define TAPP_ASSIGN_CPLX_CASE(ENUM, T, CONJFN) \
    case ENUM: ((T*)D)[index_D] = *(T_ACC_C*)accum; break;
#define TAPP_ASSIGN_PREC_CASE(T_R, T_C)                                     \
    {                                                                        \
        typedef T_R T_ACC_R; typedef T_C T_ACC_C;                           \
        switch (type_D)                                                      \
        {                                                                    \
        TAPP_REAL_TYPES(TAPP_ASSIGN_REAL_CASE)                               \
        TAPP_COMPLEX_TYPES(TAPP_ASSIGN_CPLX_CASE)                            \
        default: break;                                                      \
        }                                                                    \
    }

void assign_D(void* D, TAPP_datatype type_D, int64_t index_D, void* accum, TAPP_prectype prec)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type_D) { TAPP_DATATYPE_LIST(TAPP_ASSIGN_ID_CASE) default: break; }
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        TAPP_ASSIGN_PREC_CASE(float, complex float);
        break;
    case TAPP_F64F64_ACCUM_F64:
        TAPP_ASSIGN_PREC_CASE(double, complex double);
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        /* Matches the original: real type_D reads back the F16 accumulator;
         * complex type_D is a no-op (never assigned) here, same as before. */
        {
            typedef _Float16 T_ACC_R;
            switch (type_D)
            {
            TAPP_REAL_TYPES(TAPP_ASSIGN_REAL_CASE)
            case TAPP_C32:
            case TAPP_C64:
                break;
            default:
                break;
            }
        }
        break;
#endif
    default:
        break;
    }
}
#undef TAPP_ASSIGN_PREC_CASE
#undef TAPP_ASSIGN_CPLX_CASE
#undef TAPP_ASSIGN_REAL_CASE
#undef TAPP_ASSIGN_ID_CASE

