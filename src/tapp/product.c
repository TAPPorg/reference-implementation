/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "structs.h"
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
void C_relation_to_D(int* relation, const int64_t* idx_C, const int64_t* idx_D, int nmode_D);
void calculate_beta_C(const void* beta, const void* C, TAPP_datatype type_C, int index_C, TAPP_element_op op_C, TAPP_prectype prec, void* output, TAPP_datatype type_output);
void calculate_alpha_A_B(const void* alpha, const void* A, TAPP_datatype type_A, int index_A, TAPP_element_op op_A, const void* B, TAPP_datatype type_B, int index_B, TAPP_element_op op_B, TAPP_prectype prec, void* output, TAPP_datatype type_D);
void calculate_op_D(void* output, TAPP_datatype type_D, TAPP_element_op op_D);
void assign_D(void* D, TAPP_datatype type_D, int64_t index_D, void* val);

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

void C_relation_to_D(int* relation, const int64_t* idx_C, const int64_t* idx_D, int nmode_D) {
    for (int i = 0; i < nmode_D; i++) {
        for (int j = 0; j < nmode_D; j++) {
            if (idx_D[i] == idx_C[j]) {
                relation[i] = j;
            }
        }
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