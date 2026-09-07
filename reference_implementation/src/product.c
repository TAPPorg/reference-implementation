/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "../include/product.h"
#include "../include/product_template.h"

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
int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int unique_idx_code);
int check_extents_pair(int nmode_X, const int64_t* idx_X, const int64_t* extents_X, int nmode_Y, const int64_t* idx_Y, const int64_t* extents_Y, int missmatch_code);
int check_same_structure(int nmode_X, const int64_t* idx_X, const int64_t* extents_X, int nmode_Y, const int64_t* idx_Y, const int64_t* extents_Y, int nmode_code, int idx_code, int extent_code);

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

// malloc(0) is legitimately allowed to return NULL, so only treat a NULL
// result as fatal when a non-zero allocation was actually requested.
void* checked_malloc(size_t size) {
    if (size == 0) return NULL;
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "TAPP reference implementation: out of memory (requested %zu bytes)\n", size);
        exit(1);
    }
    return ptr;
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
    if (error_status != 0)
    {
        return error_status;
    }
    struct plan* plan_ptr = checked_malloc(sizeof(struct plan));
    
    plan_ptr->A = A;

    plan_ptr->idx_A = checked_malloc(((struct tensor_info*)A)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_A, idx_A, ((struct tensor_info*)A)->nmode * sizeof(int64_t));


    plan_ptr->B = B;

    plan_ptr->idx_B = checked_malloc(((struct tensor_info*)B)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_B, idx_B, ((struct tensor_info*)B)->nmode * sizeof(int64_t));


    plan_ptr->C = C;

    plan_ptr->idx_C = checked_malloc(((struct tensor_info*)C)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_C, idx_C, ((struct tensor_info*)C)->nmode * sizeof(int64_t));


    plan_ptr->D = D;

    plan_ptr->idx_D = checked_malloc(((struct tensor_info*)D)->nmode * sizeof(int64_t));
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

int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int unique_idx_code)
{
    for (int i = 0; i < nmode_origin; i++)
    {
        int idx_found = 0;
        for (int j = 0; j < nmode_test_A; j++)
        {
            if (idx_origin[i] == idx_test_A[j])
            {
                idx_found++;
                break;
            }
        }
        for (int j = 0; j < nmode_test_B; j++)
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
    for (int i = 0; i < nmode_X; i++)
    {
        for (int j = 0; j < nmode_Y; j++)
        {
            if (idx_X[i] == idx_Y[j] && extents_X[i] != extents_Y[j])
            {
                return missmatch_code;
            }
        }
    }
    return 0;
}

int check_same_structure(int nmode_X, const int64_t* idx_X, const int64_t* extents_X, int nmode_Y, const int64_t* idx_Y, const int64_t* extents_Y, int nmode_code, int idx_code, int extent_code)
{
    if(nmode_X != nmode_Y)
    {
        return nmode_code;
    }

    for (int i = 0; i < nmode_Y; i++)
    {
        if (idx_Y[i] != idx_X[i])
        {
            return idx_code;
        }
        if (extents_Y[i] != extents_X[i])
        {
            return extent_code;
        }
    }
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
    *H_idx_ptr = checked_malloc(max_H_nmode * sizeof(int64_t));
    int H_nmode = 0;
    for (int i = 0; i < nmode_A; i++)
    {
        bool already_handled = false;
        for (int j = 0; j < i; j++)
        {
            if (idx_A[i] == idx_A[j]) {
                already_handled = true;
                break;
            }
        }
        if (already_handled) continue;

        bool in_B = false;
        for (int j = 0; j < nmode_B; j++)
        {
            if (idx_A[i] == idx_B[j]) {
                in_B = true;
                break;
            }
        }
        if (!in_B) continue;

        bool in_D = false;
        for (int j = 0; j < nmode_D; j++)
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
    *P_idx_ptr = checked_malloc(max_P_nmode * sizeof(int64_t));
    int P_nmode = 0;
    for (int i = 0; i < nmode_A; i++)
    {
        bool already_handled = false;
        for (int j = 0; j < i; j++)
        {
            if (idx_A[i] == idx_A[j]) {
                already_handled = true;
                break;
            }
        }
        if (already_handled) continue;

        bool in_B = false;
        for (int j = 0; j < nmode_B; j++)
        {
            if (idx_A[i] == idx_B[j]) {
                in_B = true;
                break;
            }
        }
        if (!in_B) continue;

        bool in_D = false;
        for (int j = 0; j < nmode_D; j++)
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
    *FX_idx_ptr = checked_malloc(max_FX_nmode * sizeof(int64_t));
    int FX_nmode = 0;
    for (int i = 0; i < nmode_X; i++)
    {
        bool already_handled = false;
        for (int j = 0; j < i; j++)
        {
            if (idx_X[i] == idx_X[j]) {
                already_handled = true;
                break;
            }
        }
        if (already_handled) continue;

        bool in_Y = false;
        for (int j = 0; j < nmode_Y; j++)
        {
            if (idx_X[i] == idx_y[j]) {
                in_Y = true;
                break;
            }
        }
        if (in_Y) continue;

        bool in_D = false;
        for (int j = 0; j < nmode_D; j++)
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
    *IX_idx_ptr = checked_malloc(max_IX_nmode * sizeof(int64_t));
    int IX_nmode = 0;
    for (int i = 0; i < nmode_X; i++)
    {
        bool already_handled = false;
        for (int j = 0; j < i; j++)
        {
            if (idx_X[i] == idx_X[j]) {
                already_handled = true;
                break;
            }
        }
        if (already_handled) continue;

        bool in_Y = false;
        for (int j = 0; j < nmode_Y; j++)
        {
            if (idx_X[i] == idx_y[j]) {
                in_Y = true;
                break;
            }
        }
        if (in_Y) continue;

        bool in_D = false;
        for (int j = 0; j < nmode_Z; j++)
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
    *G_extents_X_ptr = checked_malloc(G_nmode * sizeof(int64_t));
    for (int i = 0; i < G_nmode; i++)
    {
        (*G_extents_X_ptr)[i] = 0;
        for (int j = 0; j < nmode_X; j++)
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
    *G_strides_X_ptr = checked_malloc(G_nmode * sizeof(int64_t));
    for (int i = 0; i < G_nmode; i++)
    {
        (*G_strides_X_ptr)[i] = 0;
        for (int j = 0; j < nmode_X; j++)
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
    for (int i = 0; i < nmode; i++)
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

typedef double complex dcomplex;
typedef float complex scomplex;

#define TAPP_EXECUTE_PRODUCT_TEMPLATE2(T_A, T_B, T_C, T_D, T_ACC) Tapp_execute_product_ ## T_A ## _ ## T_B ## _ ## T_C ## _ ## T_D ## _ ## T_ACC
#define TAPP_EXECUTE_PRODUCT_TEMPLATE(T_A, T_B, T_C, T_D, T_ACC) TAPP_EXECUTE_PRODUCT_TEMPLATE2(T_A, T_B, T_C, T_D, T_ACC)

typedef TAPP_error (*TAPP_execute_product_fn)(TAPP_tensor_product,
                                              TAPP_executor,
                                              TAPP_status*,
                                              const void*,
                                              const void*,
                                              const void*,
                                              const void*,
                                              const void*,
                                              void*);

// One entry per supported (type_A, type_B, type_C, type_D, prec) combination, pointing at
// the Tapp_execute_product_<T_A>_<T_B>_<T_C>_<T_D>_<T_ACC> instantiation that implements it.
// Not every storage-type combination is supported - only the ones listed in
// generate_product_instantiations.py.
typedef struct
{
    TAPP_datatype type_A;
    TAPP_datatype type_B;
    TAPP_datatype type_C;
    TAPP_datatype type_D;
    TAPP_prectype prec;
    TAPP_execute_product_fn fn;
} TAPP_product_dispatch_entry;

// Declares one Tapp_execute_product_<T_A>_<T_B>_<T_C>_<T_D>_<T_ACC> per supported combination,
// plus the TAPP_PRODUCT_DISPATCH[] table mapping (type_A, type_B, type_C, type_D, prec) to
// them. Generated by generate_product_instantiations.py - see that file to regenerate.
#include "product_instantiations.gen.c"

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

    for (size_t i = 0; i < TAPP_PRODUCT_DISPATCH_COUNT; i++)
    {
        const TAPP_product_dispatch_entry* entry = &TAPP_PRODUCT_DISPATCH[i];
        if (entry->type_A == plan_ptr->type_A &&
            entry->type_B == plan_ptr->type_B &&
            entry->type_C == plan_ptr->type_C &&
            entry->type_D == plan_ptr->type_D &&
            entry->prec == plan_ptr->prec)
        {
            return entry->fn(plan, exec, status, alpha, A, B, beta, C, D);
        }
    }

    return 34;
}

int64_t calcualte_offset(int64_t* coords, int nmode, int64_t* strides)
{
    int64_t index = 0;
    for (int i = 0; i < nmode; i++)
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

int check_executor_existence(TAPP_executor exec, int error_code)
{
    if(!exec) return error_code;
    intptr_t* exec_ptr= &exec; //pointer to intptr_t (TAPP_executor)
    int* eip = (int*) *exec_ptr;//dereference to get the int pointer
    if((*eip) == 1 || (*eip) == 2 ||  (*eip) == 12) return 0;
    return error_code; // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
}