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
void sum_unary_contractions(void* sum, const void* tensor, int index, TAPP_element_op op, TAPP_datatype type, TAPP_prectype prec);
void calculate_beta_C(const void* beta, TAPP_datatype type_beta, bool is_complex_beta, const void* value_C, TAPP_datatype type_C, bool is_complex_C, TAPP_element_op op_C, TAPP_prectype prec, void* accum, TAPP_datatype type_accum, bool is_complex_accum);
void calculate_beta_C_default(const void* beta, TAPP_datatype type_beta, const void* value_C, TAPP_datatype type_C, TAPP_element_op op_C, void* accum, TAPP_datatype type_accum);
void calculate_beta_C_prec(const void* beta, bool is_complex_beta, const void* value_C, bool is_complex_C, TAPP_prectype prec, void* accum, bool is_complex_accum);
void calculate_alpha_A_B(const void* alpha, TAPP_datatype type_alpha, bool is_complex_alpha, const void* sum_A, TAPP_datatype type_A, bool is_complex_A, const void* sum_B, TAPP_datatype type_B, bool is_complex_B, TAPP_prectype prec, void* accum, TAPP_datatype type_accum, bool is_complex_accum);
void calculate_alpha_A_B_default(const void* alpha, TAPP_datatype type_alpha, const void* sum_A, TAPP_datatype type_A, const void* sum_B, TAPP_datatype type_B, void* accum, TAPP_datatype type_accum);
void calculate_alpha_A_B_prec(const void* alpha, bool is_complex_alpha, const void* sum_A, bool is_complex_A, const void* sum_B, bool is_complex_B, TAPP_prectype prec, void* accum, bool is_complex_accum);
void calculate_op_D(void* accum, TAPP_datatype type_D, TAPP_element_op op_D, TAPP_prectype prec);
int calcualte_offset(int64_t* coords, int nmode, int64_t* strides);
void get_typed_value(void* val, const void* tensor, int64_t index, TAPP_datatype type, TAPP_prectype prec);
void assign_D(void* D, TAPP_datatype type_D, int64_t index_D, void* accum, TAPP_prectype prec);
int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int unique_idx_code);
int check_extents(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_D, const int64_t* idx_D, const int64_t* extents_D, int missmatch_AA_code, int missmatch_AB_code, int missmatch_AD_code);
int check_same_structure(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_code, int idx_code, int extent_code);
int check_self_aliasing(int nmode, const int64_t* extents, const int64_t* strides, int error_code);
int check_tensor_existence(const void* scalar, TAPP_datatype type, const void* tensor, int error_code);
int check_executor_existence(TAPP_executor exec, int error_code);
void merge_sort_strides(int64_t* strides, int64_t*extents, int left, int right);
void merge_strides(int64_t* strides, int64_t* extents, int left, int mid, int right);
void* alloc_accum(TAPP_prectype prec, TAPP_datatype type);
void* alloc_typed_value(TAPP_prectype prec, TAPP_datatype type);
void* create_prec_scalar(const void* scalar, TAPP_datatype type, TAPP_prectype prec);
bool is_complex(TAPP_datatype type);
void set_typed_scalar_to_zero(void* sum, TAPP_prectype prec, TAPP_datatype type, bool is_complex);
void set_typed_accum_to_zero(void* accum, TAPP_prectype prec, TAPP_datatype type, bool is_complex);
void add_to_typed_accum(void* accum, const void* value, TAPP_prectype prec, TAPP_datatype type, bool is_complex);
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
    if (error_status == 0) error_status = check_extents(info_A_ptr->nmode, idx_A, info_A_ptr->extents, info_B_ptr->nmode, idx_B, info_B_ptr->extents, info_D_ptr->nmode, idx_D, info_D_ptr->extents, 9, 1, 2);
    if (error_status == 0) error_status = check_extents(info_B_ptr->nmode, idx_B, info_B_ptr->extents, info_A_ptr->nmode, idx_A, info_A_ptr->extents, info_D_ptr->nmode, idx_D, info_D_ptr->extents, 10, 1, 3);
    if (error_status == 0) error_status = check_extents(info_D_ptr->nmode, idx_D, info_D_ptr->extents, info_A_ptr->nmode, idx_A, info_A_ptr->extents, info_B_ptr->nmode, idx_B, info_B_ptr->extents, 11, 2, 3);
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
    plan_ptr->B_nmode = extract_IX_indices(((struct tensor_info*)D)->nmode, idx_D,
                                           ((struct tensor_info*)A)->nmode, idx_A,
                                           ((struct tensor_info*)B)->nmode, idx_B,
                                           &plan_ptr->B_idx);

    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->H_nmode, plan_ptr->H_idx, &plan_ptr->H_extents);
    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->P_nmode, plan_ptr->P_idx, &plan_ptr->P_extents);
    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->FA_nmode, plan_ptr->FA_idx, &plan_ptr->FA_extents);
    extract_grouped_extents(((struct tensor_info*)B)->nmode, idx_B, ((struct tensor_info*)B)->extents, plan_ptr->FB_nmode, plan_ptr->FB_idx, &plan_ptr->FB_extents);
    extract_grouped_extents(((struct tensor_info*)A)->nmode, idx_A, ((struct tensor_info*)A)->extents, plan_ptr->IA_nmode, plan_ptr->IA_idx, &plan_ptr->IA_extents);
    extract_grouped_extents(((struct tensor_info*)B)->nmode, idx_B, ((struct tensor_info*)B)->extents, plan_ptr->IB_nmode, plan_ptr->IB_idx, &plan_ptr->IB_extents);
    extract_grouped_extents(((struct tensor_info*)D)->nmode, idx_D, ((struct tensor_info*)D)->extents, plan_ptr->B_nmode, plan_ptr->B_idx, &plan_ptr->B_extents);
    
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
    extract_grouped_strides(((struct tensor_info*)D)->nmode, idx_D, ((struct tensor_info*)D)->strides, plan_ptr->B_nmode, plan_ptr->B_idx, &plan_ptr->B_strides);

    plan_ptr->H_size = calculate_size(plan_ptr->H_extents, plan_ptr->H_nmode);
    plan_ptr->P_size = calculate_size(plan_ptr->P_extents, plan_ptr->P_nmode);
    plan_ptr->FA_size = calculate_size(plan_ptr->FA_extents, plan_ptr->FA_nmode);
    plan_ptr->FB_size = calculate_size(plan_ptr->FB_extents, plan_ptr->FB_nmode);
    plan_ptr->IA_size = calculate_size(plan_ptr->IA_extents, plan_ptr->IA_nmode);
    plan_ptr->IB_size = calculate_size(plan_ptr->IB_extents, plan_ptr->IB_nmode);
    plan_ptr->B_size = calculate_size(plan_ptr->B_extents, plan_ptr->B_nmode);

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
    free(((struct plan*)plan)->B_idx);
    free(((struct plan*)plan)->H_extents);
    free(((struct plan*)plan)->P_extents);
    free(((struct plan*)plan)->FA_extents);
    free(((struct plan*)plan)->FB_extents);
    free(((struct plan*)plan)->IA_extents);
    free(((struct plan*)plan)->IB_extents);
    free(((struct plan*)plan)->B_extents);
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
    free(((struct plan*)plan)->B_strides);
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
        void* accum_aAB = alloc_accum(plan_ptr->prec, plan_ptr->type_D);
        void* accum_bC = alloc_accum(plan_ptr->prec, plan_ptr->type_D);
        void* sum_A = alloc_typed_value(plan_ptr->prec, plan_ptr->type_A);
        void* sum_B = alloc_typed_value(plan_ptr->prec, plan_ptr->type_B);
        void* value_C = alloc_typed_value(plan_ptr->prec, plan_ptr->type_C);
        void* prec_alpha = create_prec_scalar(alpha, plan_ptr->type_D, plan_ptr->prec);
        void* prec_beta = create_prec_scalar(beta, plan_ptr->type_D, plan_ptr->prec);

        bool is_complex_A = is_complex(plan_ptr->type_A);
        bool is_complex_B = is_complex(plan_ptr->type_B);
        bool is_complex_C = is_complex(plan_ptr->type_C);
        bool is_complex_D = is_complex(plan_ptr->type_D);

        float value_zero = 0;
        bool beta_is_zero = is_equal(beta, plan_ptr->type_D, &value_zero, TAPP_F32);

        set_typed_accum_to_zero(accum_bC, plan_ptr->prec, plan_ptr->type_D, is_complex_D);

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

        int64_t* B_coords = malloc(plan_ptr->B_nmode * sizeof(int64_t));
        for (int i = 0; i < plan_ptr->B_nmode; i++) B_coords[i] = 0;

        for (int h = 0; h < plan_ptr->H_nmode; h++)
        {
            int H_offset_A = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_A);
            int H_offset_B = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_B);
            int H_offset_D = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_D);

            for (int fa = 0; fa < plan_ptr->FA_nmode; fa++)
            {
                int FA_offset_A = calcualte_offset(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_strides_A);
                int FA_offset_D = calcualte_offset(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_strides_D);

                for (int fb = 0; fb < plan_ptr->FB_nmode; fb++)
                {
                    int FB_offset_B = calcualte_offset(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_strides_B);
                    int FB_offset_D = calcualte_offset(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_strides_D);

                    set_typed_accum_to_zero(accum_aAB, plan_ptr->prec, plan_ptr->type_D, is_complex_D);

                    for (int p = 0; p < plan_ptr->P_nmode; p++)
                    {
                        int P_offset_A = calcualte_offset(P_coords, plan_ptr->P_nmode, plan_ptr->P_strides_A);
                        int P_offset_B = calcualte_offset(P_coords, plan_ptr->P_nmode, plan_ptr->P_strides_B);

                        set_typed_scalar_to_zero(sum_A, plan_ptr->prec, plan_ptr->type_A, is_complex_A);
                        for (int ia = 0; ia < plan_ptr->IA_nmode; ia++)
                        {
                            int IA_offset_A = calcualte_offset(IA_coords, plan_ptr->IA_nmode, plan_ptr->IA_strides_A);
                            int offset_A = H_offset_A + FA_offset_A + IA_offset_A + P_offset_A;
                            sum_unary_contractions(sum_A, A, offset_A, plan_ptr->op_A, plan_ptr->type_A, plan_ptr->prec);
                            increment_coordinates(IA_coords, plan_ptr->IA_nmode, plan_ptr->IA_extents);
                        }

                        set_typed_scalar_to_zero(sum_B, plan_ptr->prec, plan_ptr->type_B, is_complex_B);
                        for (int ib = 0; ib < plan_ptr->IB_nmode; ib++)
                        {
                            int IB_offset_B = calcualte_offset(IB_coords, plan_ptr->IB_nmode, plan_ptr->IB_strides_B);
                            int offset_B = H_offset_B + FB_offset_B + IB_offset_B + P_offset_B;
                            sum_unary_contractions(sum_B, B, offset_B, plan_ptr->op_B, plan_ptr->type_B, plan_ptr->prec);
                            increment_coordinates(IB_coords, plan_ptr->IB_nmode, plan_ptr->IB_extents);
                        }

                        calculate_alpha_A_B(prec_alpha, plan_ptr->type_D, is_complex_D, sum_A, plan_ptr->type_A, is_complex_A, sum_B, plan_ptr->type_B, is_complex_B, plan_ptr->prec, accum_aAB, plan_ptr->type_D, is_complex_D);

                        increment_coordinates(P_coords, plan_ptr->P_nmode, plan_ptr->P_extents);
                    }

                    for (int b = 0; b < plan_ptr->B_nmode; b++)
                    {
                        int B_offset = calcualte_offset(B_coords, plan_ptr->B_nmode, plan_ptr->B_strides);
                        int offset_D = H_offset_D + FA_offset_D + FB_offset_D + B_offset;

                        set_typed_accum_to_zero(accum_aAB, plan_ptr->prec, plan_ptr->type_D, is_complex_D);

                        add_to_typed_accum(accum, accum_aAB, plan_ptr->prec, plan_ptr->type_D, is_complex_D);

                        if (!beta_is_zero)
                        {
                            get_typed_value(value_C, C, offset_D, plan_ptr->type_C, plan_ptr->prec);
                            calculate_beta_C(prec_beta, plan_ptr->type_D, is_complex_D, value_C, plan_ptr->type_C, is_complex_C, plan_ptr->op_C, plan_ptr->prec, accum_bC, plan_ptr->type_D, is_complex_D);
                            add_to_typed_accum(accum, accum_bC, plan_ptr->prec, plan_ptr->type_D, is_complex_D);
                        }
                    
                        calculate_op_D(accum, plan_ptr->type_D, plan_ptr->op_D, plan_ptr->prec);

                        assign_D(D, plan_ptr->type_D, offset_D, accum, plan_ptr->prec);

                        increment_coordinates(B_coords, plan_ptr->B_nmode, plan_ptr->B_extents);
                    }

                    increment_coordinates(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_extents);
                }

                increment_coordinates(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_extents);
            }
            
            increment_coordinates(H_coords, plan_ptr->H_nmode, plan_ptr->H_extents);
        }

        free(accum);
        free(accum_aAB);
        free(accum_bC);
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
        free(B_coords);
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

int check_extents(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_D, const int64_t* idx_D, const int64_t* extents_D, int missmatch_AA_code, int missmatch_AB_code, int missmatch_AD_code)
{
    for (size_t i = 0; i < nmode_A; i++)
    {
        for (size_t j = 0; j < nmode_A; j++)
        {
            if (idx_A[i] == idx_A[j] && extents_A[i] != extents_A[j])
            {
                return missmatch_AA_code;
            }
        }
        for (size_t j = 0; j < nmode_B; j++)
        {
            if (idx_A[i] == idx_B[j] && extents_A[i] != extents_B[j])
            {
                return missmatch_AB_code;
            }
        }
        for (size_t j = 0; j < nmode_D; j++)
        {
            if (idx_A[i] == idx_D[j] && extents_A[i] != extents_D[j])
            {
                return missmatch_AD_code;
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

int check_self_aliasing(int nmode, const int64_t* extents, const int64_t* strides, int error_code)
{
    if (nmode <= 1)
    {
        return 0;
    }
    for (size_t i = 0; i < nmode; i++)
    {
        if (strides[i] == 0)
        {
            return error_code;
        }
    }

    int64_t* sorted_strides = malloc(nmode * sizeof(int64_t));
    int64_t* sorted_extents = malloc(nmode * sizeof(int64_t));
    for (size_t i = 0; i < nmode; i++)
    {
        sorted_strides[i] = labs(strides[i]);
        sorted_extents[i] = extents[i];
    }
    merge_sort_strides(sorted_strides, sorted_extents, 0, nmode - 1);
    int status = 0;
    for (size_t i = 0; i < nmode - 1; i++)
    {
        if (sorted_strides[i + 1] < sorted_strides[i] * sorted_extents[i])
        {
            status = error_code;
            break;
        }
    }
    free(sorted_strides);
    free(sorted_extents);
    return status;
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

void merge_sort_strides(int64_t* strides, int64_t*extents, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        merge_sort_strides(strides, extents, left, mid);
        merge_sort_strides(strides, extents, mid + 1, right);

        merge_strides(strides, extents, left, mid, right);
    }
}

void merge_strides(int64_t* strides, int64_t* extents, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int Ls[n1], Rs[n2];
    int Le[n1], Re[n2];

    for (int i = 0; i < n1; i++)
    {
        Ls[i] = strides[left + i];
        Le[i] = extents[left + i];
    }
    for (int j = 0; j < n2; j++)
    {
        Rs[j] = strides[mid + 1 + j];
        Re[j] = extents[mid + 1 + j];
    }

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (Ls[i] < Rs[j])
        {
            strides[k] = Ls[i];
            extents[k] = Le[i];
            i++;
        }
        else if (Ls[i] > Rs[j])
        {
            strides[k] = Rs[j];
            extents[k] = Re[j];
            j++;
        }
        else
        {
            if (Le[i] <= Re[j])
            {
                strides[k] = Ls[i];
                extents[k] = Le[i];
                i++;
            }
            else
            {
                strides[k] = Rs[j];
                extents[k] = Re[j];
                j++;
            }
        }
        k++;
    }

    while (i < n1)
    {
        strides[k] = Ls[i];
        extents[k] = Le[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        strides[k] = Rs[j];
        extents[k] = Re[j];
        j++;
        k++;
    }
}

void* alloc_accum(TAPP_prectype prec, TAPP_datatype type)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type)
        {
        case TAPP_F32:
            return malloc(sizeof(float));
            break;
        case TAPP_F64:
            return malloc(sizeof(double));
            break;
        case TAPP_C32:
            return malloc(sizeof(complex float));
            break;
        case TAPP_C64:
            return malloc(sizeof(complex double));
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            return malloc(sizeof(_Float16));
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            return malloc(sizeof(__bf16));
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        switch (type)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            return malloc(sizeof(float));
            break;
        case TAPP_C32:
        case TAPP_C64:
            return malloc(sizeof(complex float));
            break;
        default:
            break;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        switch (type)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            return malloc(sizeof(double));
            break;
        case TAPP_C32:
        case TAPP_C64:
            return malloc(sizeof(complex double));
            break;
        default:
            break;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        return malloc(sizeof(_Float16));
        break;
#endif
    default:
        break;
    }
    return NULL;
}

void* alloc_typed_value(TAPP_prectype prec, TAPP_datatype type)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type)
        {
        case TAPP_F32:
            return malloc(sizeof(float));
            break;
        case TAPP_F64:
            return malloc(sizeof(double));
            break;
        case TAPP_C32:
            return malloc(sizeof(complex float));
            break;
        case TAPP_C64:
            return malloc(sizeof(complex double));
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            return malloc(sizeof(_Float16));
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            return malloc(sizeof(__bf16));
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            return malloc(sizeof(float));
            break;
        case TAPP_C32:
        case TAPP_C64:
            return malloc(sizeof(complex float));
            break;
        default:
            break;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        switch (type)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            return malloc(sizeof(double));
            break;
        case TAPP_C32:
        case TAPP_C64:
            return malloc(sizeof(complex double));
            break;
        default:
            break;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
    case TAPP_F16F16_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            return malloc(sizeof(_Float16));
            break;
        case TAPP_C32:
        case TAPP_C64:
            return malloc(sizeof(complex _Float16));
            break;
        default:
            break;
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            return malloc(sizeof(__bf16));
            break;
        case TAPP_C32:
        case TAPP_C64:
            return malloc(sizeof(complex __bf16));
            break;
        default:
            break;
        }
        break;
#endif
    default:
        break;
    }
    return NULL;
}

void* create_prec_scalar(const void* scalar, TAPP_datatype type, TAPP_prectype prec)
{
    switch (type)
    {
    case TAPP_F32:
        switch (prec)
        {
        case TAPP_DEFAULT_PREC:
        case TAPP_F32F32_ACCUM_F32:
        {
            float* prec_scalar = malloc(sizeof(float));
            *prec_scalar = *(float*)scalar;
            return prec_scalar;
            break;
        }
        case TAPP_F64F64_ACCUM_F64:
        {
            double* prec_scalar = malloc(sizeof(double));
            *prec_scalar = *(float*)scalar;
            return prec_scalar;
            break;
        }
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16F16_ACCUM_F16:
        case TAPP_F16F16_ACCUM_F32:
        {
            _Float16* prec_scalar = malloc(sizeof(_Float16));
            *prec_scalar = *(float*)scalar;
            return prec_scalar;
            break;
        }
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16BF16_ACCUM_F32:
        {
            __bf16* prec_alpha = malloc(sizeof(__bf16));
            *prec_scalar = *(float*)scalar;
            return prec_scalar;
            break;
        }
#endif
        default:
            break;
        }
    case TAPP_F64:
        switch (prec)
        {
        case TAPP_F32F32_ACCUM_F32:
        {
            float* prec_scalar = malloc(sizeof(float));
            *prec_scalar = *(double*)scalar;
            return prec_scalar;
            break;
        }
        case TAPP_DEFAULT_PREC:
        case TAPP_F64F64_ACCUM_F64:
        {
            double* prec_scalar = malloc(sizeof(double));
            *prec_scalar = *(double*)scalar;
            return prec_scalar;
            break;
        }
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16F16_ACCUM_F16:
        case TAPP_F16F16_ACCUM_F32:
        {
            _Float16* prec_scalar = malloc(sizeof(_Float16));
            *prec_scalar = *(double*)scalar;
            return prec_scalar;
            break;
        }
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16BF16_ACCUM_F32:
        {
            __bf16* prec_alpha = malloc(sizeof(__bf16));
            *prec_scalar = *(double*)scalar;
            return prec_scalar;
        }
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_C32:
        switch (prec)
        {
        case TAPP_DEFAULT_PREC:
        case TAPP_F32F32_ACCUM_F32:
        {
            complex float* prec_scalar = malloc(sizeof(complex float));
            *prec_scalar = *(complex float*)scalar;
            return prec_scalar;
            break;
        }
        case TAPP_F64F64_ACCUM_F64:
        {
            complex double* prec_scalar = malloc(sizeof(complex double));
            *prec_scalar = *(complex float*)scalar;
            return prec_scalar;
            break;
        }
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16F16_ACCUM_F16:
        case TAPP_F16F16_ACCUM_F32:
        {
            complex _Float16* prec_scalar = malloc(sizeof(complex _Float16));
            *prec_scalar = *(complex float*)scalar;
            return prec_scalar;
            break;
        }
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16BF16_ACCUM_F32:
        {
            complex __bf16* prec_alpha = malloc(sizeof(complex __bf16));
            *prec_scalar = *(complex float*)scalar;
            return prec_scalar;
            break;
        }
#endif
        default:
            break;
        }
        break;
    case TAPP_C64:
        switch (prec)
        {
        case TAPP_F32F32_ACCUM_F32:
        {
            complex float* prec_scalar = malloc(sizeof(complex float));
            *prec_scalar = *(complex double*)scalar;
            return prec_scalar;
            break;
        }
        case TAPP_DEFAULT_PREC:
        case TAPP_F64F64_ACCUM_F64:
        {
            complex double* prec_scalar = malloc(sizeof(complex double));
            *prec_scalar = *(complex double*)scalar;
            return prec_scalar;
            break;
        }
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16F16_ACCUM_F16:
        case TAPP_F16F16_ACCUM_F32:
        {
            complex _Float16* prec_scalar = malloc(sizeof(complex _Float16));
            *prec_scalar = *(complex double*)scalar;
            return prec_scalar;
            break;
        }
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16BF16_ACCUM_F32:
        {
            complex __bf16* prec_alpha = malloc(sizeof(complex __bf16));
            *prec_scalar = *(complex double*)scalar;
            return prec_scalar;
            break;
        }
#endif
        default:
            break;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16:
        switch (prec)
        {
        case TAPP_F32F32_ACCUM_F32:
        {
            float* prec_scalar = malloc(sizeof(float));
            *prec_scalar = *(_Float16*)scalar;
            return prec_scalar;
            break;
        }
        case TAPP_F64F64_ACCUM_F64:
        {
            double* prec_scalar = malloc(sizeof(double));
            *prec_scalar = *(_Float16*)scalar;
            return prec_scalar;
            break;
        }
        case TAPP_DEFAULT_PREC:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16F16_ACCUM_F16:
        case TAPP_F16F16_ACCUM_F32:
        {
            _Float16* prec_scalar = malloc(sizeof(_Float16));
            *prec_scalar = *(_Float16*)scalar;
            return prec_scalar;
            break;
        }
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16BF16_ACCUM_F32:
        {
            __bf16* prec_alpha = malloc(sizeof(__bf16));
            *prec_scalar = *(_Float16*)scalar;
            return prec_scalar;
            break;
        }
#endif
        default:
            break;
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16:
        switch (prec)
        {
        case TAPP_F32F32_ACCUM_F32:
        {
            float* prec_scalar = malloc(sizeof(float));
            *prec_scalar = *(__bf16*)scalar;
            return prec_scalar;
            break;
        }
        case TAPP_F64F64_ACCUM_F64:
        {
            double* prec_scalar = malloc(sizeof(double));
            *prec_scalar = *(__bf16*)scalar;
            return prec_scalar;
            break;
        }
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16F16_ACCUM_F16:
        case TAPP_F16F16_ACCUM_F32:
        {
            _Float16* prec_scalar = malloc(sizeof(_Float16));
            *prec_scalar = *(__bf16*)scalar;
            return prec_scalar;
            break;
        }
#endif
        case TAPP_DEFAULT_PREC:
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16BF16_ACCUM_F32:
        {
            __bf16* prec_alpha = malloc(sizeof(__bf16));
            *prec_scalar = *(__bf16*)scalar;
            return prec_scalar;
            break;
        }
#endif
        default:
            break;
        }
        break;
#endif
    default:
        return false;
        break;
    }
    return NULL;
}

bool is_complex(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
    case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16:
#endif
        return false;
        break;
    case TAPP_C32:
    case TAPP_C64:
        return true;
        break;
    default:
        return false;
        break;
    }
}

void set_typed_scalar_to_zero(void* sum, TAPP_prectype prec, TAPP_datatype type, bool is_complex)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type)
        {
        case TAPP_F32:
            *((float*)sum) = 0;
            break;
        case TAPP_F64:
            *((double*)sum) = 0;
            break;
        case TAPP_C32:
            *((complex float*)sum) = 0;
            break;
        case TAPP_C64:
            *((complex double*)sum) = 0;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *((_Float16*)sum) = 0;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *((__bf16*)sum) = 0;
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
        if (is_complex)
        {
            *(complex float*)sum = 0;
        }
        else
        {
            *(float*)sum = 0;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        if (is_complex)
        {
            *(complex double*)sum = 0;
        }
        else
        {
            *(double*)sum = 0;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
    case TAPP_F16F16_ACCUM_F32:
        if (is_complex)
        {
            *(complex _Float16*)sum = 0;
        }
        else
        {
            *(_Float16*)sum = 0;
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        if (is_complex)
        {
            *(complex__bf16*)sum = 0;
        }
        else
        {
            *(__bf16*)sum = 0;
        }
        break;
#endif
    default:
        break;
    }
}

void set_typed_accum_to_zero(void* accum, TAPP_prectype prec, TAPP_datatype type, bool is_complex)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type)
        {
        case TAPP_F32:
            *((float*)accum) = 0;
            break;
        case TAPP_F64:
            *((double*)accum) = 0;
            break;
        case TAPP_C32:
            *((complex float*)accum) = 0;
            break;
        case TAPP_C64:
            *((complex double*)accum) = 0;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *((_Float16*)accum) = 0;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *((__bf16*)accum) = 0;
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        if (is_complex)
        {
            *(complex float*)accum = 0;
        }
        else
        {
            *(float*)accum = 0;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        if (is_complex)
        {
            *(complex double*)accum = 0;
        }
        else
        {
            *(double*)accum = 0;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        if (is_complex)
        {
            *(complex _Float16*)accum = 0;
        }
        else
        {
            *(_Float16*)accum = 0;
        }
        break;
#endif
    default:
        break;
    }
}

void add_to_typed_accum(void* accum, const void* value, TAPP_prectype prec, TAPP_datatype type, bool is_complex)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type)
        {
        case TAPP_F32:
            *((float*)accum) += *(float*)value;
            break;
        case TAPP_F64:
            *((double*)accum) += *(double*)value;
            break;
        case TAPP_C32:
            *((complex float*)accum) += *(complex float*)value;
            break;
        case TAPP_C64:
            *((complex double*)accum) += *(complex double*)value;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *((_Float16*)accum) += *(_Float16*)value;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *((__bf16*)accum) += *(__bf16*)value;
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        if (is_complex)
        {
            *(complex float*)accum += *(complex float*)value;
        }
        else
        {
            *(float*)accum += *(float*)value;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        if (is_complex)
        {
            *(complex double*)accum += *(complex double*)value;
        }
        else
        {
            *(double*)accum += *(double*)value;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        if (is_complex)
        {
            *(complex _Float16*)accum += *(complex _Float16*)value;
        }
        else
        {
            *(_Float16*)accum += *(_Float16*)value;
        }
        break;
#endif
    default:
        break;
    }
}

bool is_equal(const void* val, TAPP_datatype type, const void* comp_val, TAPP_datatype comp_type)
{
    switch (type)
    {
    case TAPP_F32:
        switch (comp_type)
        {
        case TAPP_F32:
            return *(float*)val == *(float*)comp_val;
            break;
        case TAPP_F64:
            return *(float*)val == *(double*)comp_val;
            break;
        case TAPP_C32:
            return *(float*)val == *(complex float*)comp_val;
            break;
        case TAPP_C64:
            return *(float*)val == *(complex double*)comp_val;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            return *(float*)val == *(_Float16*)comp_val;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            return *(float*)val == *(__bf16*)comp_val;
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F64:
        switch (comp_type)
        {
        case TAPP_F32:
            return *(double*)val == *(float*)comp_val;
            break;
        case TAPP_F64:
            return *(double*)val == *(double*)comp_val;
            break;
        case TAPP_C32:
            return *(double*)val == *(complex float*)comp_val;
            break;
        case TAPP_C64:
            return *(double*)val == *(complex double*)comp_val;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            return *(double*)val == *(_Float16*)comp_val;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            return *(double*)val == *(__bf16*)comp_val;
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_C32:
        switch (comp_type)
        {
        case TAPP_F32:
            return *(complex float*)val == *(float*)comp_val;
            break;
        case TAPP_F64:
            return *(complex float*)val == *(double*)comp_val;
            break;
        case TAPP_C32:
            return *(complex float*)val == *(complex float*)comp_val;
            break;
        case TAPP_C64:
            return *(complex float*)val == *(complex double*)comp_val;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            return *(complex float*)val == *(_Float16*)comp_val;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            return *(complex float*)val == *(__bf16*)comp_val;
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_C64:
        switch (comp_type)
        {
        case TAPP_F32:
            return *(complex double*)val == *(float*)comp_val;
            break;
        case TAPP_F64:
            return *(complex double*)val == *(double*)comp_val;
            break;
        case TAPP_C32:
            return *(complex double*)val == *(complex float*)comp_val;
            break;
        case TAPP_C64:
            return *(complex double*)val == *(complex double*)comp_val;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            return *(complex double*)val == *(_Float16*)comp_val;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            return *(complex double*)val == *(__bf16*)comp_val;
            break;
#endif
        default:
            break;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16:
        switch (comp_type)
        {
        case TAPP_F32:
            return *(_Float16*)val == *(float*)comp_val;
            break;
        case TAPP_F64:
            return *(_Float16*)val == *(double*)comp_val;
            break;
        case TAPP_C32:
            return *(_Float16*)val == *(complex float*)comp_val;
            break;
        case TAPP_C64:
            return *(_Float16*)val == *(complex double*)comp_val;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            return *(_Float16*)val == *(_Float16*)comp_val;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            return *(_Float16*)val == *(__bf16*)comp_val;
            break;
#endif
        default:
            break;
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16:
        switch (comp_type)
        {
        case TAPP_F32:
            return *(__bf16*)val == *(float*)comp_val;
            break;
        case TAPP_F64:
            return *(__bf16*)val == *(double*)comp_val;
            break;
        case TAPP_C32:
            return *(__bf16*)val == *(complex float*)comp_val;
            break;
        case TAPP_C64:
            return *(__bf16*)val == *(complex double*)comp_val;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            return *(__bf16*)val == *(_Float16*)comp_val;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            return *(__bf16*)val == *(__bf16*)comp_val;
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
    return false;
}

void sum_unary_contractions(void* sum, const void* tensor, int index, TAPP_element_op op, TAPP_datatype type, TAPP_prectype prec)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type)
        {
        case TAPP_F32:
            *(float*)sum += ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(double*)sum += ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex float*)sum += op == TAPP_CONJUGATE ? conjf(((complex float*)tensor)[index]) : ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex double*)sum += op == TAPP_CONJUGATE ? conj(((complex double*)tensor)[index]) : ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(_Float16*)sum += ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(__bf16*)sum += ((__bf16*)tensor)[index];
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
            *(float*)sum += ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(float*)sum += ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex float*)sum += op == TAPP_CONJUGATE ? conjf(((complex float*)tensor)[index]) : ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex float*)sum += op == TAPP_CONJUGATE ? conj(((complex double*)tensor)[index]) : ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(float*)sum += ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(float*)sum += ((__bf16*)tensor)[index];
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        switch (type)
        {
        case TAPP_F32:
            *(double*)sum += ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(double*)sum += ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex double*)sum += op == TAPP_CONJUGATE ? conjf(((complex float*)tensor)[index]) : ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex double*)sum += op == TAPP_CONJUGATE ? conj(((complex double*)tensor)[index]) : ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(double*)sum += ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(double*)sum += ((__bf16*)tensor)[index];
            break;
#endif
        default:
            break;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
    case TAPP_F16F16_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
            *(_Float16*)sum += ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(_Float16*)sum += ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex _Float16*)sum += ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex _Float16*)sum += ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(_Float16*)sum += ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(_Float16*)sum += ((__bf16*)tensor)[index];
            break;
#endif
        default:
            break;
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
            *(__bf16*)sum += ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(__bf16*)sum += ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex __bf16*)sum += ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex __bf16*)sum += ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(__bf16*)sum += ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(__bf16*)sum += ((__bf16*)tensor)[index];
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

void calculate_beta_C(const void* beta, TAPP_datatype type_beta, bool is_complex_beta, const void* value_C, TAPP_datatype type_C, bool is_complex_C, TAPP_element_op op_C, TAPP_prectype prec, void* accum, TAPP_datatype type_accum, bool is_complex_accum)
{
    if (prec == TAPP_DEFAULT_PREC)
    {
        calculate_beta_C_default(beta, type_beta, value_C, type_C, op_C, accum, type_accum);
    }
    else
    {
        calculate_beta_C_prec(beta, is_complex_beta, value_C, is_complex_C, prec, accum, is_complex_accum);
    }
}

void calculate_beta_C_default(const void* beta, TAPP_datatype type_beta, const void* value_C, TAPP_datatype type_C, TAPP_element_op op_C, void* accum, TAPP_datatype type_accum)
{
    switch (type_accum)
    {
    case TAPP_F32:
        switch (type_C)
        {
        case TAPP_F32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(float*)accum = *(float*)beta * *(float*)value_C;
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * *(float*)value_C;
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(float*)value_C;
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(float*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(float*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(float*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(float*)accum = *(float*)beta * *(double*)value_C;
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * *(double*)value_C;
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(double*)value_C;
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(double*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(double*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(double*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(float*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(float*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(float*)accum = *(float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(_Float16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(_Float16*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(float*)accum = *(float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(__bf16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(__bf16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(__bf16*)value_C;
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
        switch (type_C)
        {
        case TAPP_F32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(double*)accum = *(float*)beta * *(float*)value_C;
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * *(float*)value_C;
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(float*)value_C;
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(float*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(float*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(float*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(double*)accum = *(float*)beta * *(double*)value_C;
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * *(double*)value_C;
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(double*)value_C;
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(double*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(double*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(double*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(double*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(double*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(double*)accum = *(float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(_Float16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(_Float16*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(double*)accum = *(float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(__bf16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(__bf16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(__bf16*)value_C;
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
        switch (type_C)
        {
        case TAPP_F32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex float*)accum = *(float*)beta * *(float*)value_C;
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(float*)value_C;
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(float*)value_C;
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(float*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(float*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(float*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex float*)accum = *(float*)beta * *(double*)value_C;
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(double*)value_C;
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(double*)value_C;
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(double*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(double*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(double*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex float*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex float*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex float*)accum = *(float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(_Float16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(_Float16*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex float*)accum = *(float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(__bf16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(__bf16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(__bf16*)value_C;
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
        switch (type_C)
        {
        case TAPP_F32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex double*)accum = *(float*)beta * *(float*)value_C;
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(float*)value_C;
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(float*)value_C;
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(float*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(float*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(float*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex double*)accum = *(float*)beta * *(double*)value_C;
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(double*)value_C;
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(double*)value_C;
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(double*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(double*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(double*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex double*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex double*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex double*)accum = *(float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(_Float16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(_Float16*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(complex double*)accum = *(float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(__bf16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(__bf16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(__bf16*)value_C;
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
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16:
        switch (type_C)
        {
        case TAPP_F32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(_Float16*)accum = *(float*)beta * *(float*)value_C;
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(float*)value_C;
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(float*)value_C;
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(float*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(float*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(float*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(_Float16*)accum = *(float*)beta * *(double*)value_C;
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(double*)value_C;
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(double*)value_C;
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(double*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(double*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(double*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(_Float16*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(_Float16*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(_Float16*)accum = *(float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(_Float16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(_Float16*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(_Float16*)accum = *(float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(__bf16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(__bf16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(__bf16*)value_C;
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
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16:
        switch (type_C)
        {
        case TAPP_F32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(__bf16*)accum = *(float*)beta * *(float*)value_C;
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(float*)value_C;
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(float*)value_C;
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(float*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(float*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(float*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(__bf16*)accum = *(float*)beta * *(double*)value_C;
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(double*)value_C;
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(double*)value_C;
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(double*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(double*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(double*)value_C;
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_beta)
            {
            case TAPP_F32:
                *(__bf16*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)value_C) : *(complex float*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_beta)
            {
            case TAPP_F32:
                *(__bf16*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)value_C) : *(complex double*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(__bf16*)accum = *(float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(_Float16*)value_C;
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(_Float16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(_Float16*)value_C);
                break;
#endif
            default:
                break;
            }
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_beta)
            {
            case TAPP_F32:
                *(__bf16*)accum = *(float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(__bf16*)value_C;
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(__bf16*)value_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(__bf16*)value_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(__bf16*)value_C;
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

void calculate_beta_C_prec(const void* beta, bool is_complex_beta, const void* value_C, bool is_complex_C, TAPP_prectype prec, void* accum, bool is_complex_accum)
{
    switch (prec)
    {
    case TAPP_F32F32_ACCUM_F32:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex float*)beta * *(complex float*)value_C;
                }
                else
                {
                    *(complex float*)accum = *(float*)beta * *(complex float*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex float*)beta * *(float*)value_C;
                }
                else
                {
                    *(complex float*)accum = *(float*)beta * *(float*)value_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex float*)beta * *(complex float*)value_C;
                }
                else
                {
                    *(float*)accum = *(float*)beta * *(complex float*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex float*)beta * *(float*)value_C;
                }
                else
                {
                    *(float*)accum = *(float*)beta * *(float*)value_C;
                }
            }
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex double*)accum = *(complex double*)beta * *(complex double*)value_C;
                }
                else
                {
                    *(complex double*)accum = *(double*)beta * *(complex double*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex double*)accum = *(complex double*)beta * *(double*)value_C;
                }
                else
                {
                    *(complex double*)accum = *(double*)beta * *(double*)value_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(double*)accum = *(complex double*)beta * *(complex double*)value_C;
                }
                else
                {
                    *(double*)accum = *(double*)beta * *(complex double*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(double*)accum = *(complex double*)beta * *(double*)value_C;
                }
                else
                {
                    *(double*)accum = *(double*)beta * *(double*)value_C;
                }
            }
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex _Float16*)accum = *(complex _Float16*)beta * *(complex _Float16*)value_C;
                }
                else
                {
                    *(complex _Float16*)accum = *(_Float16*)beta * *(complex _Float16*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex _Float16*)accum = *(complex _Float16*)beta * *(_Float16*)value_C;
                }
                else
                {
                    *(complex _Float16*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(_Float16*)accum = *(complex _Float16*)beta * *(complex _Float16*)value_C;
                }
                else
                {
                    *(_Float16*)accum = *(_Float16*)beta * *(complex _Float16*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(_Float16*)accum = *(complex _Float16*)beta * *(_Float16*)value_C;
                }
                else
                {
                    *(_Float16*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                }
            }
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex _Float16*)beta * *(complex _Float16*)value_C;
                }
                else
                {
                    *(complex float*)accum = *(_Float16*)beta * *(complex _Float16*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex _Float16*)beta * *(_Float16*)value_C;
                }
                else
                {
                    *(complex float*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex _Float16*)beta * *(complex _Float16*)value_C;
                }
                else
                {
                    *(float*)accum = *(_Float16*)beta * *(complex _Float16*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex _Float16*)beta * *(_Float16*)value_C;
                }
                else
                {
                    *(float*)accum = *(_Float16*)beta * *(_Float16*)value_C;
                }
            }
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex __bf16*)beta * *(complex __bf16*)value_C;
                }
                else
                {
                    *(complex float*)accum = *(__bf16*)beta * *(complex __bf16*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex __bf16*)beta * *(__bf16*)value_C;
                }
                else
                {
                    *(complex float*)accum = *(__bf16*)beta * *(__bf16*)value_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex __bf16*)beta * *(complex __bf16*)value_C;
                }
                else
                {
                    *(float*)accum = *(__bf16*)beta * *(complex __bf16*)value_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex __bf16*)beta * *(__bf16*)value_C;
                }
                else
                {
                    *(float*)accum = *(__bf16*)beta * *(__bf16*)value_C;
                }
            }
        }
        break;
#endif
    default:
        break;
    }
}

void calculate_alpha_A_B(const void* alpha, TAPP_datatype type_alpha, bool is_complex_alpha, const void* sum_A, TAPP_datatype type_A, bool is_complex_A, const void* sum_B, TAPP_datatype type_B, bool is_complex_B, TAPP_prectype prec, void* accum, TAPP_datatype type_accum, bool is_complex_accum)
{
    if (prec == TAPP_DEFAULT_PREC)
    {
        calculate_alpha_A_B_default(alpha, type_alpha, sum_A, type_A, sum_B, type_B, accum, type_accum);
    }
    else
    {
        calculate_alpha_A_B_prec(alpha, is_complex_alpha, sum_A, is_complex_A, sum_B, is_complex_B, prec, accum, is_complex_accum);
    }
}

void calculate_alpha_A_B_default(const void* alpha, TAPP_datatype type_alpha, const void* sum_A, TAPP_datatype type_A, const void* sum_B, TAPP_datatype type_B, void* accum, TAPP_datatype type_accum)
{
    switch (type_accum)
    {
    case TAPP_F32:
        switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
    case TAPP_F64:
        switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
    case TAPP_C32:
        switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
    case TAPP_C64:
        switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16:
        switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16:
        switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
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
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
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
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            switch (type_B)
            {
            case TAPP_F32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_F64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C32:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
            case TAPP_C64:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
#endif
                default:
                    break;
                }
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                switch (type_alpha)
                {
                case TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#ifdef TAPP_REFERENCE_ENABLE_F16
                case TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
                case TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
#endif
    default:
        break;
    }
}

void calculate_alpha_A_B_prec(const void* alpha, bool is_complex_alpha, const void* sum_A, bool is_complex_A, const void* sum_B, bool is_complex_B, TAPP_prectype prec, void* accum, bool is_complex_accum)
{
    switch (prec)
    {
    case TAPP_F32F32_ACCUM_F32:
        if (is_complex_accum)
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    }
                }
            }
        }
        else
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    }
                }
            }
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        if (is_complex_accum)
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    }
                    else
                    {
                        *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    }
                    else
                    {
                        *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    }
                    else
                    {
                        *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    }
                    else
                    {
                        *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    }
                }
            }
        }
        else
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    }
                    else
                    {
                        *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    }
                    else
                    {
                        *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    }
                    else
                    {
                        *(double*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    }
                    else
                    {
                        *(double*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    }
                }
            }
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        if (is_complex_accum)
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex _Float16*)accum += *(complex _Float16*)alpha * *(complex _Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                    else
                    {
                        *(complex _Float16*)accum += *(_Float16*)alpha * *(complex _Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex _Float16*)accum += *(complex _Float16*)alpha * *(complex _Float16*)sum_A * *(_Float16*)sum_B;
                    }
                    else
                    {
                        *(complex _Float16*)accum += *(_Float16*)alpha * *(complex _Float16*)sum_A * *(_Float16*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex _Float16*)accum += *(complex _Float16*)alpha * *(_Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                    else
                    {
                        *(complex _Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex _Float16*)accum += *(complex _Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    }
                    else
                    {
                        *(complex _Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    }
                }
            }
        }
        else
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(_Float16*)accum += *(complex _Float16*)alpha * *(complex _Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                    else
                    {
                        *(_Float16*)accum += *(_Float16*)alpha * *(complex _Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(_Float16*)accum += *(complex _Float16*)alpha * *(complex _Float16*)sum_A * *(_Float16*)sum_B;
                    }
                    else
                    {
                        *(_Float16*)accum += *(_Float16*)alpha * *(complex _Float16*)sum_A * *(_Float16*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(_Float16*)accum += *(complex _Float16*)alpha * *(_Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                    else
                    {
                        *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(_Float16*)accum += *(complex _Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    }
                    else
                    {
                        *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    }
                }
            }
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
        if (is_complex_accum)
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex _Float16*)alpha * *(complex _Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(_Float16*)alpha * *(complex _Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex _Float16*)alpha * *(complex _Float16*)sum_A * *(_Float16*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(_Float16*)alpha * *(complex _Float16*)sum_A * *(_Float16*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex _Float16*)alpha * *(_Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex _Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    }
                }
            }
        }
        else
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex _Float16*)alpha * *(complex _Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(_Float16*)alpha * *(complex _Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex _Float16*)alpha * *(complex _Float16*)sum_A * *(_Float16*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(_Float16*)alpha * *(complex _Float16*)sum_A * *(_Float16*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex _Float16*)alpha * *(_Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex _Float16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex _Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    }
                }
            }
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        if (is_complex_accum)
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex __bf16*)alpha * *(complex __bf16*)sum_A * *(complex __bf16*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(__bf16*)alpha * *(complex __bf16*)sum_A * *(complex __bf16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex __bf16*)alpha * *(complex __bf16*)sum_A * *(__bf16*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(__bf16*)alpha * *(complex __bf16*)sum_A * *(__bf16*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex __bf16*)alpha * *(__bf16*)sum_A * *(complex __bf16*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex __bf16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(complex float*)accum += *(complex __bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    }
                    else
                    {
                        *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    }
                }
            }
        }
        else
        {
            if (is_complex_A)
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex __bf16*)alpha * *(complex __bf16*)sum_A * *(complex __bf16*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(__bf16*)alpha * *(complex __bf16*)sum_A * *(complex __bf16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex __bf16*)alpha * *(complex __bf16*)sum_A * *(__bf16*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(__bf16*)alpha * *(complex __bf16*)sum_A * *(__bf16*)sum_B;
                    }
                }
            }
            else
            {
                if (is_complex_B)
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex __bf16*)alpha * *(__bf16*)sum_A * *(complex __bf16*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex __bf16*)sum_B;
                    }
                }
                else
                {
                    if (is_complex_alpha)
                    {
                        *(float*)accum += *(complex __bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    }
                    else
                    {
                        *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    }
                }
            }
        }
        break;
#endif
    default:
        break;
    }
}

void calculate_op_D(void* accum, TAPP_datatype type_D, TAPP_element_op op_D, TAPP_prectype prec)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type_D)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            break;
        case TAPP_C32:
            if (op_D == TAPP_CONJUGATE)
            {
                *(complex float*)accum = conjf(*(complex float*)accum);
            }
            break;
        case TAPP_C64:
            if (op_D == TAPP_CONJUGATE)
            {
                *(complex double*)accum = conj(*(complex double*)accum);
            }
            break;
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        switch (type_D)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            break;
        case TAPP_C32:
        case TAPP_C64:
            if (op_D == TAPP_CONJUGATE)
            {
                *(complex float*)accum = conjf(*(complex float*)accum);
            }
            break;
        default:
            break;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        switch (type_D)
        {
        case TAPP_F32:
        case TAPP_F64:
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
#endif
            break;
        case TAPP_C32:
        case TAPP_C64:
            if (op_D == TAPP_CONJUGATE)
            {
                *(complex double*)accum = conjf(*(complex double*)accum);
            }
            break;
        default:
            break;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        break;
#endif
    default:
        break;
    }
}

void get_typed_value(void* val, const void* tensor, int64_t index, TAPP_datatype type, TAPP_prectype prec)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type)
        {
        case TAPP_F32:
            *(float*)val = ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(double*)val = ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex float*)val = ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex double*)val = ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(_Float16*)val = ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(__bf16*)val = ((__bf16*)tensor)[index];
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
            *(float*)val = ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(float*)val = ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex float*)val = ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex float*)val = ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(float*)val = ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(float*)val = ((__bf16*)tensor)[index];
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        switch (type)
        {
        case TAPP_F32:
            *(double*)val = ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(double*)val = ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex double*)val = ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex double*)val = ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(double*)val = ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(double*)val = ((__bf16*)tensor)[index];
            break;
#endif
        default:
            break;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
    case TAPP_F16F16_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
            *(_Float16*)val = ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(_Float16*)val = ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex _Float16*)val = ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex _Float16*)val = ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(_Float16*)val = ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(_Float16*)val = ((__bf16*)tensor)[index];
            break;
#endif
        default:
            break;
        }
        break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
        switch (type)
        {
        case TAPP_F32:
            *(__bf16*)val = ((float*)tensor)[index];
            break;
        case TAPP_F64:
            *(__bf16*)val = ((double*)tensor)[index];
            break;
        case TAPP_C32:
            *(complex __bf16*)val = ((complex float*)tensor)[index];
            break;
        case TAPP_C64:
            *(complex __bf16*)val = ((complex double*)tensor)[index];
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            *(__bf16*)val = ((_Float16*)tensor)[index];
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            *(__bf16*)val = ((__bf16*)tensor)[index];
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

void assign_D(void* D, TAPP_datatype type_D, int64_t index_D, void* accum, TAPP_prectype prec)
{
    switch (prec)
    {
    case TAPP_DEFAULT_PREC:
        switch (type_D)
        {
        case TAPP_F32:
            ((float*)D)[index_D] = *(float*)accum;
            break;
        case TAPP_F64:
            ((double*)D)[index_D] = *(double*)accum;
            break;
        case TAPP_C32:
            ((complex float*)D)[index_D] = *(complex float*)accum;
            break;
        case TAPP_C64:
            ((complex double*)D)[index_D] = *(complex double*)accum;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            ((_Float16*)D)[index_D] = *(_Float16*)accum;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            ((__bf16*)D)[index_D] = *(__bf16*)accum;
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F32F32_ACCUM_F32:
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F32:
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
    case TAPP_BF16BF16_ACCUM_F32:
#endif
        switch (type_D)
        {
        case TAPP_F32:
            ((float*)D)[index_D] = *(float*)accum;
            break;
        case TAPP_F64:
            ((double*)D)[index_D] = *(float*)accum;
            break;
        case TAPP_C32:
            ((complex float*)D)[index_D] = *(complex float*)accum;
            break;
        case TAPP_C64:
            ((complex double*)D)[index_D] = *(complex float*)accum;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            ((_Float16*)D)[index_D] = *(float*)accum;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            ((__bf16*)D)[index_D] = *(float*)accum;
            break;
#endif
        default:
            break;
        }
        break;
    case TAPP_F64F64_ACCUM_F64:
        switch (type_D)
        {
        case TAPP_F32:
            ((float*)D)[index_D] = *(double*)accum;
            break;
        case TAPP_F64:
            ((double*)D)[index_D] = *(double*)accum;
            break;
        case TAPP_C32:
            ((complex float*)D)[index_D] = *(complex double*)accum;
            break;
        case TAPP_C64:
            ((complex double*)D)[index_D] = *(complex double*)accum;
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            ((_Float16*)D)[index_D] = *(double*)accum;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            ((__bf16*)D)[index_D] = *(double*)accum;
            break;
#endif
        default:
            break;
        }
        break;
#ifdef TAPP_REFERENCE_ENABLE_F16
    case TAPP_F16F16_ACCUM_F16:
        switch (type_D)
        {
        case TAPP_F32:
            ((float*)D)[index_D] = *(_Float16*)accum;
            break;
        case TAPP_F64:
            ((double*)D)[index_D] = *(_Float16*)accum;
            break;
        case TAPP_C32:
            break;
        case TAPP_C64:
            break;
#ifdef TAPP_REFERENCE_ENABLE_F16
        case TAPP_F16:
            ((_Float16*)D)[index_D] = *(_Float16*)accum;
            break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
        case TAPP_BF16:
            ((__bf16*)D)[index_D] = *(_Float16*)accum;
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
