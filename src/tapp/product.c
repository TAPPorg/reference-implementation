/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "tapp_ex_imp.h"
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __x86_64__
    typedef _Float16 float16;
#elif __arm__
    typedef _fp16 float16;
#endif

int calculate_contracted_indices(int nmode_A, int nmode_B, int nmode_D, const int64_t* idx_A, const int64_t* idx_B, const int64_t* idx_D, int64_t* idx_contraction);
void calculate_contracted_extents(int contractions, int64_t* idx_contraction, int nmode, const int64_t* idx, int64_t* extents, int64_t* extents_contraction);
void compile_strides(int64_t* strides, int ndim, const int64_t* idx, int ndim_D, const int64_t* idx_D, int contractions, int64_t* idx_contraction, int64_t* free_strides, int64_t* contracted_strides);
int64_t calculate_size(int64_t* extents, int nmode);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
void zero_array(int64_t* arr, int size);
void calculate_beta_C(const void* beta, const void* C, TAPP_datatype type_C, int index_C, TAPP_element_op op_C, TAPP_prectype prec, void* output, TAPP_datatype type_output);
void calculate_alpha_A_B(const void* alpha, const void* A, TAPP_datatype type_A, int index_A, TAPP_element_op op_A, const void* B, TAPP_datatype type_B, int index_B, TAPP_element_op op_B, TAPP_prectype prec, void* output, TAPP_datatype type_D);
void calculate_op_D(void* output, TAPP_datatype type_D, TAPP_element_op op_D);
void assign_D(void* D, TAPP_datatype type_D, int64_t index_D, void* val);
int check_repeated_idx(int nmode, const int64_t* idx, int error_code);
int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int no_occurrence_code, int unknown_operation_code);
int check_einsum(int nmode_A, const int64_t* idx_A, int nmode_B, const int64_t* idx_B, int nmode_D, const int64_t* idx_D);
int check_extents(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_D, const int64_t* idx_D, const int64_t* extents_D, int missmatch_AB_code, int missmatch_AD_code);
int check_same_structure(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_code, int idx_code, int extent_code);
int check_self_aliasing(int nmode, const int64_t* extents, const int64_t* strides, int error_code);
void merge_sort_strides(int64_t* strides, int64_t*extents, int left, int right);
void merge_strides(int64_t* strides, int64_t* extents, int left, int mid, int right);


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
                                      TAPP_prectype prec) {
    struct plan* plan_ptr = malloc(sizeof(struct plan));
    plan_ptr->handle = handle;

    plan_ptr->op_A = op_A;
    plan_ptr->A = A;
    
    plan_ptr->idx_A = malloc(((struct tensor_info*)A)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_A, idx_A, ((struct tensor_info*)A)->nmode * sizeof(int64_t));


    plan_ptr->op_B = op_B;
    plan_ptr->B = B;
    
    plan_ptr->idx_B = malloc(((struct tensor_info*)B)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_B, idx_B, ((struct tensor_info*)B)->nmode * sizeof(int64_t));
    

    plan_ptr->op_C = op_C;
    plan_ptr->C = C;
    
    plan_ptr->idx_C = malloc(((struct tensor_info*)C)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_C, idx_C, ((struct tensor_info*)C)->nmode * sizeof(int64_t));


    plan_ptr->op_D = op_D;
    plan_ptr->D = D;
    
    plan_ptr->idx_D = malloc(((struct tensor_info*)D)->nmode * sizeof(int64_t));
    memcpy(plan_ptr->idx_D, idx_D, ((struct tensor_info*)D)->nmode * sizeof(int64_t));

    plan_ptr->prec = prec;

    *plan = (TAPP_tensor_product)plan_ptr;

    return 0;
}

TAPP_error TAPP_destory_tensor_product(TAPP_tensor_product plan) {
    free(((struct plan*)plan)->idx_A);
    free(((struct plan*)plan)->idx_B);
    free(((struct plan*)plan)->idx_C);
    free(((struct plan*)plan)->idx_D);
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
                                void* D) {
    struct plan* plan_ptr = (struct plan*)plan;
    TAPP_handle handle = plan_ptr->handle;

    TAPP_element_op op_A = plan_ptr->op_A;
    TAPP_tensor_info A_info = (TAPP_tensor_info)(plan_ptr->A);
    struct tensor_info* A_info_ptr = (struct tensor_info*)(plan_ptr->A);
    const int64_t* idx_A = plan_ptr->idx_A;

    TAPP_element_op op_B = plan_ptr->op_B;
    TAPP_tensor_info B_info = (TAPP_tensor_info)(plan_ptr->B);
    struct tensor_info* B_info_ptr = (struct tensor_info*)(plan_ptr->B);
    const int64_t* idx_B = plan_ptr->idx_B;

    TAPP_element_op op_C = plan_ptr->op_C;
    TAPP_tensor_info C_info = (TAPP_tensor_info)(plan_ptr->C);
    struct tensor_info* C_info_ptr = (struct tensor_info*)(plan_ptr->C);
    const int64_t* idx_C = plan_ptr->idx_C;

    TAPP_element_op op_D = plan_ptr->op_D;
    TAPP_tensor_info D_info = (TAPP_tensor_info)(plan_ptr->D);
    struct tensor_info* D_info_ptr = (struct tensor_info*)(plan_ptr->D);
    const int64_t* idx_D = plan_ptr->idx_D;
    
    TAPP_prectype prec = plan_ptr->prec;

    TAPP_datatype type_A = A_info_ptr->type;
    int nmode_A = TAPP_get_nmodes(A_info);
    int64_t* extents_A = malloc(nmode_A * sizeof(int64_t));
    TAPP_get_extents(A_info, extents_A);
    int64_t* strides_A = malloc(nmode_A * sizeof(int64_t));
    TAPP_get_strides(A_info, strides_A);

    TAPP_datatype type_B = B_info_ptr->type;
    int nmode_B = TAPP_get_nmodes(B_info);
    int64_t* extents_B = malloc(nmode_B * sizeof(int64_t));
    TAPP_get_extents(B_info, extents_B);
    int64_t* strides_B = malloc(nmode_B * sizeof(int64_t));
    TAPP_get_strides(B_info, strides_B);

    TAPP_datatype type_C = C_info_ptr->type;
    int nmode_C = TAPP_get_nmodes(C_info);
    int64_t* extents_C = malloc(nmode_C * sizeof(int64_t));
    TAPP_get_extents(C_info, extents_C);
    int64_t* strides_C = malloc(nmode_C * sizeof(int64_t));
    TAPP_get_strides(C_info, strides_C);

    TAPP_datatype type_D = D_info_ptr->type;
    int nmode_D = TAPP_get_nmodes(D_info);
    int64_t* extents_D = malloc(nmode_D * sizeof(int64_t));
    TAPP_get_extents(D_info, extents_D);
    int64_t* strides_D = malloc(nmode_D * sizeof(int64_t));
    TAPP_get_strides(D_info, strides_D);

    int error_status = 0;

    error_status = check_repeated_idx(nmode_A, idx_A, 1);
    error_status = error_status == 0 ? check_repeated_idx(nmode_B, idx_B, 2) : error_status;
    error_status = error_status == 0 ? check_repeated_idx(nmode_D, idx_D, 3) : error_status;

    error_status = error_status == 0 ? check_einsum(nmode_A, idx_A, nmode_B, idx_B, nmode_D, idx_D) : error_status;
    error_status = error_status == 0 ? check_extents(nmode_A, idx_A, extents_A, nmode_B, idx_B, extents_B, nmode_D, idx_D, extents_D, 4, 5) : error_status;
    error_status = error_status == 0 ? check_extents(nmode_B, idx_B, extents_B, nmode_A, idx_A, extents_A, nmode_D, idx_D, extents_D, 4, 6) : error_status;
    error_status = error_status == 0 ? check_extents(nmode_D, idx_D, extents_D, nmode_A, idx_A, extents_A, nmode_B, idx_B, extents_B, 5, 6) : error_status;
    error_status = error_status == 0 ? check_same_structure(nmode_C, idx_C, extents_C, nmode_D, idx_D, extents_D, 11, 12, 13) : error_status;
    error_status = error_status == 0 ? check_self_aliasing(nmode_D, extents_D, strides_D, 14) : error_status;
    if (error_status != 0) {
        free(extents_A);
        free(strides_A);
        free(extents_B);
        free(strides_B);
        free(extents_C);
        free(strides_C);
        free(extents_D);
        free(strides_D);
        return error_status;
    } 

    int contractions = (nmode_A + nmode_B - nmode_D) / 2;
    int64_t* idx_contraction = malloc(contractions * sizeof(int64_t));
    contractions = calculate_contracted_indices(nmode_A, nmode_B, nmode_D, idx_A, idx_B, idx_D, idx_contraction);
    idx_contraction = (int64_t*)realloc(idx_contraction, contractions * sizeof(int64_t));

    int64_t* extents_contraction = malloc(contractions * sizeof(int64_t));
    calculate_contracted_extents(contractions, idx_contraction, nmode_A, idx_A, extents_A, extents_contraction);

    int size_contraction = calculate_size(extents_contraction, contractions);

    int64_t* free_strides_A = malloc(nmode_D * sizeof(int64_t));
    int64_t* contracted_strides_A = malloc(contractions * sizeof(int64_t));
    compile_strides(strides_A, nmode_A, idx_A, nmode_D, idx_D, contractions, idx_contraction, free_strides_A, contracted_strides_A);

    int64_t* free_strides_B = malloc(nmode_D * sizeof(int64_t));
    int64_t* contracted_strides_B = malloc(contractions * sizeof(int64_t));
    compile_strides(strides_B, nmode_B, idx_B, nmode_D, idx_D, contractions, idx_contraction, free_strides_B, contracted_strides_B);

    int64_t* coordinates_D = malloc(nmode_D * sizeof(int64_t));
    zero_array(coordinates_D, nmode_D);
    int64_t* coordinates_contraction = malloc(contractions * sizeof(int64_t));
    zero_array(coordinates_contraction, contractions);

    int64_t size_D = calculate_size(extents_D, nmode_D);

    void* val;

    switch (type_D)
    {
    case TAPP_F32:
        val = malloc(sizeof(float));
        break;
    case TAPP_F64:
        val = malloc(sizeof(double));
        break;
    case TAPP_C32:
        val = malloc(sizeof(float complex));
        break;
    case TAPP_C64:
        val = malloc(sizeof(double complex));
        break;
    case TAPP_F16:
        val = malloc(sizeof(float16));
        break;
    case TAPP_BF16:
        //val = malloc(sizeof(__bf16));
        break;
    default:
        break;
    }

    for (int i = 0; i < size_D; i++) {
        int index_A_free = 0; // Index calculated from free indices of A
        int index_B_free = 0; // Index calculated from free indices of B
        int index_C = 0;
        int index_D = 0;
        for (int j = 0; j < nmode_D; j++) {
            index_A_free += coordinates_D[j] * free_strides_A[j];
            index_B_free += coordinates_D[j] * free_strides_B[j];
            index_C += coordinates_D[j] * strides_C[j];
            index_D += coordinates_D[j] * strides_D[j];
        }
        calculate_beta_C(beta, C, type_C, index_C, op_C, prec, val, type_D);
        for (int j = 0; j < size_contraction; j++) {
            int index_A = index_A_free;
            int index_B = index_B_free;
            for (int i = 0; i < contractions; i++)
            {
                index_A += coordinates_contraction[i] * contracted_strides_A[i];
                index_B += coordinates_contraction[i] * contracted_strides_B[i];
            }
            calculate_alpha_A_B(alpha, A, type_A, index_A, op_A, B, type_B, index_B, op_B, prec, val, type_D);
            increment_coordinates(coordinates_contraction, contractions, extents_contraction);
        }
        calculate_op_D(val, type_D, op_D);
        assign_D(D, type_D, index_D, val);
        increment_coordinates(coordinates_D, nmode_D, extents_D);
    }

    free(val);
    free(extents_A);
    free(strides_A);
    free(extents_B);
    free(strides_B);
    free(extents_C);
    free(strides_C);
    free(extents_D);
    free(strides_D);
    free(idx_contraction);
    free(extents_contraction);
    free(free_strides_A);
    free(contracted_strides_A);
    free(free_strides_B);
    free(contracted_strides_B);
    free(coordinates_D);
    free(coordinates_contraction);
    return 0;
}

int calculate_contracted_indices(int nmode_A, int nmode_B, int nmode_D, const int64_t* idx_A, const int64_t* idx_B, const int64_t* idx_D, int64_t* idx_contraction) {
    int k = 0;
    for (int i = 0; i < nmode_A; i++) {
        bool index_found_in_B = false;
        bool index_found_in_D = false;
        for (int j = 0; j < nmode_B; j++) {
            if (idx_A[i] == idx_B[j]) {
                index_found_in_B = true;
                break;
            }
        }
        for (int j = 0; j < nmode_D; j++) {
            if (idx_A[i] == idx_D[j]) {
                index_found_in_D = true;
                break;
            }
        }
        if (index_found_in_B && !index_found_in_D) {
            idx_contraction[k] = idx_A[i];
            k++;
        }
    }
    return k;
}

void calculate_contracted_extents(int contractions, int64_t* idx_contraction, int nmode, const int64_t* idx, int64_t* extents, int64_t* extents_contraction) {
    for (int i = 0; i < contractions; i++) {
        for (int j = 0; j < nmode; j++) {
            if (idx_contraction[i] == idx[j]) {
                extents_contraction[i] = extents[j];
            }
        }
    }
}

void compile_strides(int64_t* strides, int nmode, const int64_t* idx, int nmode_D, const int64_t* idx_D, int contractions, int64_t* idx_contraction, int64_t* free_strides, int64_t* contracted_strides) {
    // Calculate strides for free indices
    for (int i = 0; i < nmode_D; i++) {
        bool index_found = false;
        for (int j = 0; j < nmode; j++) {
            if (idx_D[i] == idx[j]) {
                free_strides[i] = strides[j];
                index_found = true;
            }
        }
        if (!index_found) {
            free_strides[i] = 0;
        }
    }

    // Calculate strides for contracted indices
    for (int i = 0; i < contractions; i++) {
        for (int j = 0; j < nmode; j++) {
            if (idx_contraction[i] == idx[j]) {
                contracted_strides[i] = strides[j];
            }
        }
    }
}

int64_t calculate_size(int64_t* extents, int nmode) {
    int size = 1;
    for (int i = 0; i < nmode; i++) {
        size *= extents[i];
    }
    return size;
}

void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents) {
    if (nmode <= 0) {
        return;
    }

    int k = 0;
    do
    {
        coordinates[k] = (coordinates[k] + 1) % extents[k];
        k++;
    } while (coordinates[k - 1] == 0 && k < nmode);
}

bool compare_arrays(int* arr_a, int* arr_b, int size) {
    for (int i = 0; i < size; i++) {
        if (arr_a[i] != arr_b[i]) {
            return false;
        }
    }
    return true;
}

void zero_array(int64_t* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = 0;
    }
}

int check_repeated_idx(int nmode, const int64_t* idx, int error_code) {
    for (size_t i = 0; i < nmode; i++) {
        int count = 0;
        for (size_t j = 0; j < nmode; j++) {
            if (idx[i] == idx[j]) {
                count++;
            }
        }
        if (count != 1) {
            return error_code;
        }
    }
    return 0;
}

int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int no_occurrence_code, int unknown_operation_code) {
    int double_occurrence_count = 0;
    for (size_t i = 0; i < nmode_origin; i++) {
        int idx_found = 0;
        for (size_t j = 0; j < nmode_test_A; j++)
        {
            if (idx_origin[i] == idx_test_A[j]) {
                idx_found++;
                break;
            }
        }
        for (size_t j = 0; j < nmode_test_B; j++)
        {
            if (idx_origin[i] == idx_test_B[j]) {
                idx_found++;
                break;
            }
        }
        if (idx_found == 0) { //No other occurrence, unused idx, error
            return no_occurrence_code;
        }
        else if (idx_found == 2)
        {
            double_occurrence_count++;
        }
    }

    if (double_occurrence_count == 0) { //Not hadamard, no error
        return 0;
    }
    else if (double_occurrence_count == nmode_origin) { //Might be hadamard, no error
        return 1;
    }
    else { //Both singel and double occurrence, neither hadamard nor contraction, error
        return unknown_operation_code;
    }
}

int check_einsum(int nmode_A, const int64_t* idx_A, int nmode_B, const int64_t* idx_B, int nmode_D, const int64_t* idx_D) {
    int status_A = check_idx_occurrence(nmode_A, idx_A, nmode_B, idx_B, nmode_D, idx_D, -7, -10);
    if (status_A < 0) {
        return -status_A;
    }

    int status_B = check_idx_occurrence(nmode_B, idx_B, nmode_A, idx_A, nmode_D, idx_D, -8, -10);
    if (status_B < 0) {
        return -status_B;
    }

    int status_D = check_idx_occurrence(nmode_D, idx_D, nmode_A, idx_A, nmode_B, idx_B, -9, -10);
    if (status_D < 0) {
        return -status_D;
    }
    
    if (status_A != status_B || status_A != status_D) {
        return 10;
    }
    return 0;
}

int check_extents(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_D, const int64_t* idx_D, const int64_t* extents_D, int missmatch_AB_code, int missmatch_AD_code) {
    for (size_t i = 0; i < nmode_A; i++)
    {
        for (size_t j = 0; j < nmode_B; j++)
        {
            if (idx_A[i] == idx_B[j] && extents_A[i] != extents_B [j]) {
                return missmatch_AB_code;
            }
        }
        for (size_t j = 0; j < nmode_D; j++)
        {
            if (idx_A[i] == idx_D[j] && extents_A[i] != extents_D [j]) {
                return missmatch_AD_code;
            }
        }
    }
    return 0;
}

int check_same_structure(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_code, int idx_code, int extent_code) {
    if(nmode_A != nmode_B) {
        return nmode_code;
    }

    for (size_t i = 0; i < nmode_B; i++)
    {
        if (idx_B[i] != idx_A[i]) {
            return idx_code;
        }
        if (extents_B[i] != extents_A[i]) {
            return extent_code;
        }
    }
    return 0;
}

int check_self_aliasing(int nmode, const int64_t* extents, const int64_t* strides, int error_code) {
    if (nmode <= 1) {
        return 0;
    }
    for (size_t i = 0; i < nmode; i++)
    {
        if (strides[i] == 0) {
            return error_code;
        }
    }
    
    int64_t* sorted_strides = malloc(nmode * sizeof(int64_t));
    int64_t* sorted_extents = malloc(nmode * sizeof(int64_t));
    for (size_t i = 0; i < nmode; i++)
    {
        sorted_strides[i] = abs(strides[i]);
        sorted_extents[i] = extents[i];
    }
    merge_sort_strides(sorted_strides, sorted_extents, 0, nmode - 1);
    int status = 0;
    for (size_t i = 0; i < nmode - 1; i++)
    {
        if (sorted_strides[i + 1] < sorted_strides[i] * sorted_extents[i]) {
            status = error_code;
            break;
        }
    }
    free(sorted_strides);
    free(sorted_extents);
    return status;
}

void merge_sort_strides(int64_t* strides, int64_t*extents, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        merge_sort_strides(strides, extents, left, mid);
        merge_sort_strides(strides, extents, mid + 1, right);
        
        merge_strides(strides, extents, left, mid, right);
    }
}

void merge_strides(int64_t* strides, int64_t* extents, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    int Ls[n1], Rs[n2];
    int Le[n1], Re[n2];

    for (int i = 0; i < n1; i++) {
        Ls[i] = strides[left + i];
        Le[i] = extents[left + i];
    }
    for (int j = 0; j < n2; j++) {
        Rs[j] = strides[mid + 1 + j];
        Re[j] = extents[mid + 1 + j];
    }
    
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (Ls[i] <= Rs[j]) {
            strides[k] = Ls[i];
            extents[k] = Le[i];
            i++;
        } else {
            strides[k] = Rs[j];
            extents[k] = Re[i];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        strides[k] = Ls[i];
        extents[k] = Le[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        strides[k] = Rs[j];
        extents[k] = Re[j];
        j++;
        k++;
    }
}

void calculate_beta_C(const void* beta, const void* C, TAPP_datatype type_C, int index_C, TAPP_element_op op_C, TAPP_prectype prec, void* output, TAPP_datatype type_output) {
    switch (type_output)
    {
    case TAPP_F32:
        switch (type_C)
        {
        case TAPP_F32:
            *((float*)output) = *((float*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float*)C)[index_C]) : ((float*)C)[index_C]);
            break;
        case TAPP_F64:
            *((float*)output) = *((double*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double*)C)[index_C]) : ((double*)C)[index_C]);
            break;
        case TAPP_C32:
            *((float*)output) = *((float _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float _Complex*)C)[index_C]) : ((float _Complex*)C)[index_C]);
            break;
        case TAPP_C64:
            *((float*)output) = *((double _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double _Complex*)C)[index_C]) : ((double _Complex*)C)[index_C]);
            break;
        case TAPP_F16:
            *((float*)output) = *((float16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float16*)C)[index_C]) : ((float16*)C)[index_C]);
            break;
        case TAPP_BF16:
            //*((float*)output) = *((__bf16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((__bf16*)C)[index_C]) : ((__bf16*)C)[index_C]);
            break;
        default:
            break;
        }
        break;
    case TAPP_F64:
        switch (type_C)
        {
        case TAPP_F32:
            *((double*)output) = *((float*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float*)C)[index_C]) : ((float*)C)[index_C]);
            break;
        case TAPP_F64:
            *((double*)output) = *((double*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double*)C)[index_C]) : ((double*)C)[index_C]);
            break;
        case TAPP_C32:
            *((double*)output) = *((float _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float _Complex*)C)[index_C]) : ((float _Complex*)C)[index_C]);
            break;
        case TAPP_C64:
            *((double*)output) = *((double _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double _Complex*)C)[index_C]) : ((double _Complex*)C)[index_C]);
            break;
        case TAPP_F16:
            *((double*)output) = *((float16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float16*)C)[index_C]) : ((float16*)C)[index_C]);
            break;
        case TAPP_BF16:
            //*((double*)output) = *((__bf16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((__bf16*)C)[index_C]) : ((__bf16*)C)[index_C]);
            break;
        default:
            break;
        }
        break;
    case TAPP_C32:
        switch (type_C)
        {
        case TAPP_F32:
            *((float complex*)output) = *((float*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float*)C)[index_C]) : ((float*)C)[index_C]);
            break;
        case TAPP_F64:
            *((float complex*)output) = *((double*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double*)C)[index_C]) : ((double*)C)[index_C]);
            break;
        case TAPP_C32:
            *((float complex*)output) = *((float _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float _Complex*)C)[index_C]) : ((float _Complex*)C)[index_C]);
            break;
        case TAPP_C64:
            *((float complex*)output) = *((double _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double _Complex*)C)[index_C]) : ((double _Complex*)C)[index_C]);
            break;
        case TAPP_F16:
            *((float complex*)output) = *((float16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float16*)C)[index_C]) : ((float16*)C)[index_C]);
            break;
        case TAPP_BF16:
            //*((float complex*)output) = *((__bf16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((__bf16*)C)[index_C]) : ((__bf16*)C)[index_C]);
            break;
        default:
            break;
        }
        break;
    case TAPP_C64:
        switch (type_C)
        {
        case TAPP_F32:
            *((double _Complex*)output) = *((float*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float*)C)[index_C]) : ((float*)C)[index_C]);
            break;
        case TAPP_F64:
            *((double _Complex*)output) = *((double*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double*)C)[index_C]) : ((double*)C)[index_C]);
            break;
        case TAPP_C32:
            *((double _Complex*)output) = *((float _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float _Complex*)C)[index_C]) : ((float _Complex*)C)[index_C]);
            break;
        case TAPP_C64:
            *((double _Complex*)output) = *((double _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double _Complex*)C)[index_C]) : ((double _Complex*)C)[index_C]);
            break;
        case TAPP_F16:
            *((double _Complex*)output) = *((float16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float16*)C)[index_C]) : ((float16*)C)[index_C]);
            break;
        case TAPP_BF16:
            //*((double _Complex*)output) = *((__bf16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((__bf16*)C)[index_C]) : ((__bf16*)C)[index_C]);
            break;
        default:
            break;
        }
        break;
    case TAPP_F16:
        switch (type_C)
        {
        case TAPP_F32:
            *((float16*)output) = *((float*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float*)C)[index_C]) : ((float*)C)[index_C]);
            break;
        case TAPP_F64:
            *((float16*)output) = *((double*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double*)C)[index_C]) : ((double*)C)[index_C]);
            break;
        case TAPP_C32:
            *((float16*)output) = *((float _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float _Complex*)C)[index_C]) : ((float _Complex*)C)[index_C]);
            break;
        case TAPP_C64:
            *((float16*)output) = *((double _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double _Complex*)C)[index_C]) : ((double _Complex*)C)[index_C]);
            break;
        case TAPP_F16:
            *((float16*)output) = *((float16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float16*)C)[index_C]) : ((float16*)C)[index_C]);
            break;
        case TAPP_BF16:
            //*((__fp16*)output) = *((__bf16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((__bf16*)C)[index_C]) : ((__bf16*)C)[index_C]);
            break;
        default:
            break;
        }
        break;
    case TAPP_BF16:
        /*switch (type_C)
        {
        case TAPP_F32:
            *((__bf16*)output) = *((float*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float*)C)[index_C]) : ((float*)C)[index_C]);
            break;
        case TAPP_F64:
            *((__bf16*)output) = *((double*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double*)C)[index_C]) : ((double*)C)[index_C]);
            break;
        case TAPP_C32:
            *((__bf16*)output) = *((float _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((float _Complex*)C)[index_C]) : ((float _Complex*)C)[index_C]);
            break;
        case TAPP_C64:
            *((__bf16*)output) = *((double _Complex*)beta) * (op_C == TAPP_CONJUGATE ? conj(((double _Complex*)C)[index_C]) : ((double _Complex*)C)[index_C]);
            break;
        case TAPP_F16:
            *((__bf16*)output) = *((__fp16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((__fp16*)C)[index_C]) : ((__fp16*)C)[index_C]);
            break;
        case TAPP_BF16:
            *((__bf16*)output) = *((__bf16*)beta) * (op_C == TAPP_CONJUGATE ? conjf(((__bf16*)C)[index_C]) : ((__bf16*)C)[index_C]);
            break;
        default:
            break;
        }
        break;*/
    default:
        break;
    }
}

void calculate_alpha_A_B(const void* alpha, const void* A, TAPP_datatype type_A, int index_A, TAPP_element_op op_A, const void* B, TAPP_datatype type_B, int index_B, TAPP_element_op op_B, TAPP_prectype prec, void* output, TAPP_datatype type_D) {
    switch (type_D)
    {
    case TAPP_F32:
        switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                *((float*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_B)
            {
            case TAPP_F32:
                *((float*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_B)
            {
            case TAPP_F32:
                *((float*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_B)
            {
            case TAPP_F32:
                *((float*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                *((float*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float*)output) += *((__fp16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__fp16*)A)[index_A]) : ((__fp16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_BF16:
            /*switch (type_B)
            {
            case TAPP_F32:
                *((float*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
            break;
            case TAPP_C64:
                *((float*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((float*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }*/
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
                *((double*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_B)
            {
            case TAPP_F32:
                *((double*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_B)
            {
            case TAPP_F32:
                *((double*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_B)
            {
            case TAPP_F32:
                *((double*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                *((double*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double*)output) += *((__fp16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__fp16*)A)[index_A]) : ((__fp16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_BF16:
            /*switch (type_B)
            {
            case TAPP_F32:
                *((double*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((double*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }*/
            break;
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
                *((float complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_B)
            {
            case TAPP_F32:
                *((float complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_B)
            {
            case TAPP_F32:
                *((float complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_B)
            {
            case TAPP_F32:
                *((float complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                *((float complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((float complex*)output) += *((__fp16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__fp16*)A)[index_A]) : ((__fp16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_BF16:
            /*switch (type_B)
            {
            case TAPP_F32:
                *((float complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((float complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }*/
            break;
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
                *((double _Complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double _Complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double _Complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double _Complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double _Complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double _Complex*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_B)
            {
            case TAPP_F32:
                *((double _Complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double _Complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double _Complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double _Complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double _Complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double _Complex*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_B)
            {
            case TAPP_F32:
                *((double _Complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double _Complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double _Complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double _Complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double _Complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double _Complex*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_B)
            {
            case TAPP_F32:
                *((double _Complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double _Complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double _Complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double _Complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double _Complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double _Complex*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                *((double _Complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double _Complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double _Complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double _Complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double _Complex*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((double _Complex*)output) += *((__fp16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__fp16*)A)[index_A]) : ((__fp16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_BF16:
            /*switch (type_B)
            {
            case TAPP_F32:
                *((double _Complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((double _Complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((double _Complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((double _Complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((double _Complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((double _Complex*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }*/
            break;
        default:
            break;
        }
        break;
    case TAPP_F16:
        switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                *((float16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((__fp16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_B)
            {
            case TAPP_F32:
                *((float16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((__fp16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_B)
            {
            case TAPP_F32:
                *((float16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((__fp16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_B)
            {
            case TAPP_F32:
                *((float16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((__fp16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                *((float16*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((float16*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((float16*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((float16*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((float16*)output) += *((float16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float16*)A)[index_A]) : ((float16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float16*)B)[index_B]) : ((float16*)B)[index_B]);
                break;
            case TAPP_BF16:
                //*((__fp16*)output) += *((__fp16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__fp16*)A)[index_A]) : ((__fp16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        }
        break;
    case TAPP_BF16:
        /*switch (type_A)
        {
        case TAPP_F32:
            switch (type_B)
            {
            case TAPP_F32:
                *((__bf16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((__bf16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((__bf16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((__bf16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((__bf16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((__bf16*)output) += *((float*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float*)A)[index_A]) : ((float*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F64:
            switch (type_B)
            {
            case TAPP_F32:
                *((__bf16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((__bf16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((__bf16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((__bf16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((__bf16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((__bf16*)output) += *((double*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double*)A)[index_A]) : ((double*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C32:
            switch (type_B)
            {
            case TAPP_F32:
                *((__bf16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((__bf16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((__bf16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((__bf16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((__bf16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((__bf16*)output) += *((float _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((float _Complex*)A)[index_A]) : ((float _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_C64:
            switch (type_B)
            {
            case TAPP_F32:
                *((__bf16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((__bf16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((__bf16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((__bf16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((__bf16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((__bf16*)output) += *((double _Complex*)alpha) * (op_A == TAPP_CONJUGATE ? conj(((double _Complex*)A)[index_A]) : ((double _Complex*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_F16:
            switch (type_B)
            {
            case TAPP_F32:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        case TAPP_BF16:
            switch (type_B)
            {
            case TAPP_F32:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float*)B)[index_B]) : ((float*)B)[index_B]);
                break;
            case TAPP_F64:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double*)B)[index_B]) : ((double*)B)[index_B]);
                break;
            case TAPP_C32:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((float _Complex*)B)[index_B]) : ((float _Complex*)B)[index_B]);
                break;
            case TAPP_C64:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conj(((double _Complex*)B)[index_B]) : ((double _Complex*)B)[index_B]);
                break;
            case TAPP_F16:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__fp16*)B)[index_B]) : ((__fp16*)B)[index_B]);
                break;
            case TAPP_BF16:
                *((__bf16*)output) += *((__bf16*)alpha) * (op_A == TAPP_CONJUGATE ? conjf(((__bf16*)A)[index_A]) : ((__bf16*)A)[index_A]) * (op_B == TAPP_CONJUGATE ? conjf(((__bf16*)B)[index_B]) : ((__bf16*)B)[index_B]);
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }*/
        break;
    default:
        break;
    }
}

void calculate_op_D(void* output, TAPP_datatype type_D, TAPP_element_op op_D) {
    switch (type_D)
    {
    case TAPP_F32:
        if (op_D == TAPP_CONJUGATE) {
            *((float*)output) = conjf(*((float*)output));
        }
        break;
    case TAPP_F64:
        if (op_D == TAPP_CONJUGATE) {
            *((double*)output) = conj(*((double*)output));
        }
        break;
    case TAPP_C32:
        if (op_D == TAPP_CONJUGATE) {
            *((float complex*)output) = conjf(*((float complex*)output));
        }
        break;
    case TAPP_C64:
        if (op_D == TAPP_CONJUGATE) {
            *((double complex*)output) = conj(*((double complex*)output));
        }
        break;
    case TAPP_F16:
        if (op_D == TAPP_CONJUGATE) {
            *((float16*)output) = conjf(*((float16*)output));
        }
        break;
    case TAPP_BF16:
        /*if (op_D == TAPP_CONJUGATE) {
            *((__bf16*)output) = conjf(*((__bf16*)output));
        }*/
        break;
    default:
        break;
    }
}

void assign_D(void* D, TAPP_datatype type_D, int64_t index_D, void* val) {
    switch (type_D)
    {
    case TAPP_F32:
        ((float*)D)[index_D] = *((float*)val);
        break;
    case TAPP_F64:
        ((double*)D)[index_D] = *((double*)val);
        break;
    case TAPP_C32:
        ((float complex*)D)[index_D] = *((float complex*)val);
        break;
    case TAPP_C64:
        ((double complex*)D)[index_D] = *((double complex*)val);
        break;
    case TAPP_F16:
        ((float16*)D)[index_D] = *((float16*)val);
        break;
    case TAPP_BF16:
        //((__bf16*)D)[index_D] = *((__bf16*)val);
        break;
    default:
        break;
    }
}