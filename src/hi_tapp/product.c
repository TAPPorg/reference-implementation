/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "hi_tapp/tapp_ex_imp.h"
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAS_TBLIS
#include "tblis_bind.h"
#endif

int extract_binary_contractions_indices(int nmode_A, int nmode_B, int nmode_D, const int64_t* idx_A, const int64_t* idx_B, const int64_t* idx_D, int64_t** idx_contraction_ptr);
int extract_unary_contracted_indices(int nmode, int64_t* idx, int nmode_1, int64_t* idx_1, int nmode_2, int64_t* idx_2, int64_t** idx_unary_contractions_ptr);
void extract_extents(int nr_extents, int64_t* idx_extraction, int nmode, const int64_t* idx, int64_t* extents, int64_t** extracted_extents_ptr);
void compile_strides(int64_t* strides, int ndim, const int64_t* idx, int ndim_D, const int64_t* idx_D, int contractions, int64_t* idx_contraction, int64_t* free_strides, int64_t* contracted_strides);
int64_t calculate_size(int64_t* extents, int nmode);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
void zero_array(int64_t* arr, int size);
void extract_free_strides(int nmode, const int64_t* idx, int64_t* strides, int nmode_D, const int64_t* idx_D, int64_t** strides_free_ptr);
void extract_contracted_strides(int nmode, const int64_t* idx, int64_t* strides, int contractions, int64_t* idx_contraction, int64_t** strides_contractions_ptr);
void sum_unary_contractions(void* sum, const void* tensor, int index, HI_TAPP_element_op op, HI_TAPP_datatype type, HI_TAPP_prectype prec);
void calculate_beta_C(const void* beta, HI_TAPP_datatype type_beta, bool is_complex_beta, const void* val_C, HI_TAPP_datatype type_C, bool is_complex_C, HI_TAPP_element_op op_C, HI_TAPP_prectype prec, void* accum, HI_TAPP_datatype type_accum, bool is_complex_accum);
void calculate_beta_C_default(const void* beta, HI_TAPP_datatype type_beta, const void* val_C, HI_TAPP_datatype type_C, HI_TAPP_element_op op_C, void* accum, HI_TAPP_datatype type_accum);
void calculate_beta_C_prec(const void* beta, bool is_complex_beta, const void* val_C, bool is_complex_C, HI_TAPP_prectype prec, void* accum, bool is_complex_accum);
void calculate_alpha_A_B(const void* alpha, HI_TAPP_datatype type_alpha, bool is_complex_alpha, const void* sum_A, HI_TAPP_datatype type_A, bool is_complex_A, const void* sum_B, HI_TAPP_datatype type_B, bool is_complex_B, HI_TAPP_prectype prec, void* accum, HI_TAPP_datatype type_accum, bool is_complex_accum);
void calculate_alpha_A_B_default(const void* alpha, HI_TAPP_datatype type_alpha, const void* sum_A, HI_TAPP_datatype type_A, const void* sum_B, HI_TAPP_datatype type_B, void* accum, HI_TAPP_datatype type_accum);
void calculate_alpha_A_B_prec(const void* alpha, bool is_complex_alpha, const void* sum_A, bool is_complex_A, const void* sum_B, bool is_complex_B, HI_TAPP_prectype prec, void* accum, bool is_complex_accum);
void calculate_op_D(void* accum, HI_TAPP_datatype type_D, HI_TAPP_element_op op_D, HI_TAPP_prectype prec);
void get_val(void* val, const void* tensor, int64_t index, HI_TAPP_datatype type, HI_TAPP_prectype prec);
void assign_D(void* D, HI_TAPP_datatype type_D, int64_t index_D, void* accum, HI_TAPP_prectype prec);
int check_repeated_idx(int nmode, const int64_t* idx, int error_code);
int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int unique_idx_code);
int check_extents(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_D, const int64_t* idx_D, const int64_t* extents_D, int missmatch_AA_code, int missmatch_AB_code, int missmatch_AD_code);
int check_same_structure(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_code, int idx_code, int extent_code);
int check_self_aliasing(int nmode, const int64_t* extents, const int64_t* strides, int error_code);
int check_tensor_existence(const void* scalar, HI_TAPP_datatype type, const void* tensor, int error_code);
int check_executor_existence(HI_TAPP_executor exec, int error_code);
void merge_sort_strides(int64_t* strides, int64_t*extents, int left, int right);
void merge_strides(int64_t* strides, int64_t* extents, int left, int mid, int right);
void* alloc_accum(HI_TAPP_prectype prec, HI_TAPP_datatype type);
void* alloc_val(HI_TAPP_prectype prec, HI_TAPP_datatype type);
void* create_prec_scalar(const void* scalar, HI_TAPP_datatype type, HI_TAPP_prectype prec);
void* alloc_alpha(HI_TAPP_prectype prec, HI_TAPP_datatype type);
void* alloc_beta(HI_TAPP_prectype prec, HI_TAPP_datatype type);
bool is_complex(HI_TAPP_datatype type);
void zero_sum(void* sum, HI_TAPP_prectype prec, HI_TAPP_datatype type, bool is_complex);
void zero_accum(void* accum, HI_TAPP_prectype prec, HI_TAPP_datatype type, bool is_complex);
bool is_equal(const void* val, HI_TAPP_datatype type, const void* comp_val, HI_TAPP_datatype comp_type);
void compress_repeated_indices(int* nmode, int64_t** idx, int64_t** extents, int64_t** strides);
void print_tensor_(int nmode, int64_t* extents, int64_t* strides, void* data, HI_TAPP_datatype type);


HI_TAPP_error HI_TAPP_create_tensor_product(HI_TAPP_tensor_product* plan,
                                      HI_TAPP_handle handle,
                                      HI_TAPP_element_op op_A,
                                      HI_TAPP_tensor_info A,
                                      const int64_t* idx_A,
                                      HI_TAPP_element_op op_B,
                                      HI_TAPP_tensor_info B,
                                      const int64_t* idx_B,
                                      HI_TAPP_element_op op_C,
                                      HI_TAPP_tensor_info C,
                                      const int64_t* idx_C,
                                      HI_TAPP_element_op op_D,
                                      HI_TAPP_tensor_info D,
                                      const int64_t* idx_D,
                                      HI_TAPP_prectype prec)
{
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

    *plan = (HI_TAPP_tensor_product)plan_ptr;

    return 0;
}

HI_TAPP_error HI_TAPP_destory_tensor_product(HI_TAPP_tensor_product plan)
{
    free(((struct plan*)plan)->idx_A);
    free(((struct plan*)plan)->idx_B);
    free(((struct plan*)plan)->idx_C);
    free(((struct plan*)plan)->idx_D);
    free((struct plan*)plan);

    return 0;
}

HI_TAPP_error HI_TAPP_execute_product(HI_TAPP_tensor_product plan,
                                HI_TAPP_executor exec,
                                HI_TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* B,
                                const void* beta,
                                const void* C,
                                void* D)
{
   TAPP_error err;
   err =  TAPP_execute_product((TAPP_tensor_product)plan,
                                (TAPP_executor)exec,
                                (TAPP_status*)status,
                                alpha,
                                A,
                                B,
                                beta,
                                C,
                                D);
    return (HI_TAPP_error)err;
}

void print_tensor_(int nmode, int64_t* extents, int64_t* strides, void* data_, HI_TAPP_datatype type) {
    
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
        switch (type) { // HI_TAPP_datatype
          case HI_TAPP_F32:
            float* datas = (float*) data_;
            printf("%.3f", datas[index]);
            break;
          case HI_TAPP_F64:
            double* datad = (double*) data_;
            printf("%.3f", datad[index]);
            break;
          case HI_TAPP_C32:
            float complex* datac = (float complex*) data_;
            printf("%.3f+%.3fi", crealf(datac[index]), cimagf(datac[index]));
            break;
          case HI_TAPP_C64:
            double complex* dataz = (double complex*) data_;
            printf("%.3f+%.3fi", creal(dataz[index]), cimag(dataz[index]));
            break;
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

int extract_binary_contractions_indices(int nmode_A, int nmode_B, int nmode_D, const int64_t* idx_A, const int64_t* idx_B, const int64_t* idx_D, int64_t** idx_contraction_ptr)
{
    int binary_contractions = 0;
    int64_t* idx_contraction = malloc(((nmode_A + nmode_B - nmode_D) / 2)*sizeof(int64_t)); //Allocate for worst case
    for (int i = 0; i < nmode_A; i++)
    {
        bool index_found_in_B = false;
        for (int j = 0; j < nmode_B; j++)
        {
            if (idx_A[i] == idx_B[j]) 
            {
                index_found_in_B = true;
                break;
            }
        }
        if (!index_found_in_B) continue;
        bool index_found_in_D = false;
        for (int j = 0; j < nmode_D; j++)
        {
            if (idx_A[i] == idx_D[j])
            {
                index_found_in_D = true;
                break;
            }
        }
        if (index_found_in_D) continue;
        idx_contraction[binary_contractions] = idx_A[i];
        binary_contractions++;
    }
    idx_contraction = realloc(idx_contraction, binary_contractions * sizeof(int64_t)); //Reallocate for right amount
    *idx_contraction_ptr = idx_contraction;
    return binary_contractions;
}

void extract_extents(int nr_extents, int64_t* idx_extraction, int nmode, const int64_t* idx, int64_t* extents, int64_t** extracted_extents_ptr)
{
    int64_t* extracted_extents = malloc(nr_extents * sizeof(int64_t));
    for (int i = 0; i < nr_extents; i++)
    {
        for (int j = 0; j < nmode; j++)
        {
            if (idx_extraction[i] == idx[j])
            {
                extracted_extents[i] = extents[j];
            }
        }
    }
    *extracted_extents_ptr = extracted_extents;
}

int extract_unary_contracted_indices(int nmode, int64_t* idx, int nmode_1, int64_t* idx_1, int nmode_2, int64_t* idx_2, int64_t** idx_unary_contractions_ptr)
{
    int unary_contractions = 0;
    int64_t* idx_unary_contractions = malloc(nmode * sizeof(int64_t));
    for (size_t i = 0; i < nmode; i++)
    {
        bool found = false;
        for (size_t j = 0; j < nmode_1; j++)
        {
            if (idx[i] == idx_1[j])
            {
                found = true;
                break;
            }
        }
        if (found) continue;
        for (size_t j = 0; j < nmode_2; j++)
        {
            if (idx[i] == idx_2[j])
            {
                found = true;
                break;
            }
        }
        if (found) continue;
        idx_unary_contractions[unary_contractions] = idx[i];
        unary_contractions++;
    }
    idx_unary_contractions = realloc(idx_unary_contractions, unary_contractions * sizeof(int64_t));
    *idx_unary_contractions_ptr = idx_unary_contractions;
    return unary_contractions;
}

void extract_free_strides(int nmode, const int64_t* idx, int64_t* strides, int nmode_D, const int64_t* idx_D, int64_t** strides_free_ptr)
{
    int64_t* strides_free = malloc(nmode_D * sizeof(int64_t));
    for (int i = 0; i < nmode_D; i++)
    {
        bool index_found = false;
        for (int j = 0; j < nmode; j++)
        {
            if (idx_D[i] == idx[j])
            {
                strides_free[i] = strides[j];
                index_found = true;
            }
        }
        if (!index_found)
        {
            strides_free[i] = 0;
        }
    }
    *strides_free_ptr = strides_free;
}

void extract_contracted_strides(int nmode, const int64_t* idx, int64_t* strides, int contractions, int64_t* idx_contraction, int64_t** strides_contractions_ptr)
{
    int64_t* strides_contractions = malloc(contractions * sizeof(int64_t));
    for (int i = 0; i < contractions; i++)
    {
        for (int j = 0; j < nmode; j++)
        {
            if (idx_contraction[i] == idx[j])
            {
                strides_contractions[i] = strides[j];
            }
        }
    }
    *strides_contractions_ptr = strides_contractions;
}

void compile_strides(int64_t* strides, int nmode, const int64_t* idx, int nmode_D, const int64_t* idx_D, int contractions, int64_t* idx_contraction, int64_t* free_strides, int64_t* contracted_strides)
{
    // Calculate strides for free indices
    for (int i = 0; i < nmode_D; i++)
    {
        bool index_found = false;
        for (int j = 0; j < nmode; j++)
        {
            if (idx_D[i] == idx[j])
            {
                free_strides[i] = strides[j];
                index_found = true;
            }
        }
        if (!index_found)
        {
            free_strides[i] = 0;
        }
    }

    // Calculate strides for contracted indices
    for (int i = 0; i < contractions; i++)
    {
        for (int j = 0; j < nmode; j++)
        {
            if (idx_contraction[i] == idx[j])
            {
                contracted_strides[i] = strides[j];
            }
        }
    }
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

bool compare_arrays(int* arr_a, int* arr_b, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (arr_a[i] != arr_b[i])
        {
            return false;
        }
    }
    return true;
}

void zero_array(int64_t* arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = 0;
    }
}

int check_repeated_idx(int nmode, const int64_t* idx, int error_code)
{
    for (size_t i = 0; i < nmode; i++)
    {
        int count = 0;
        for (size_t j = 0; j < nmode; j++)
        {
            if (idx[i] == idx[j])
            {
                count++;
            }
        }
        if (count != 1)
        {
            return error_code;
        }
    }
    return 0;
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
        sorted_strides[i] = abs(strides[i]);
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

int check_tensor_existence(const void* scalar, HI_TAPP_datatype type, const void* tensor, int error_code)
{
    float val_zero = 0;
    return tensor == NULL && !is_equal(scalar, type, &val_zero, HI_TAPP_F32) ? error_code : 0;
}

int check_executor_existence(HI_TAPP_executor exec, int error_code)
{
    if(!exec) return error_code;
    intptr_t* exec_ptr= &exec; //pointer to intptr_t (HI_TAPP_executor)
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

void compress_repeated_indices(int* nmode, int64_t** idx, int64_t** extents, int64_t** strides)
{
    int new_nmode = 0;
    int64_t* new_idx = malloc(*nmode * sizeof(int64_t));
    int64_t* new_extents = malloc(*nmode * sizeof(int64_t));
    int64_t* new_strides = malloc(*nmode * sizeof(int64_t));
    for (size_t i = 0; i < *nmode; i++)
    {
        bool found = false;
        for (size_t j = 0; j < new_nmode; j++)
        {
            if ((*idx)[i] == new_idx[j])
            {
                found = true;
            }
        }
        if (!found)
        {
            new_idx[new_nmode] = (*idx)[i];
            new_extents[new_nmode] = (*extents)[i];
            new_strides[new_nmode] = 0;
            for (size_t j = 0; j < *nmode; j++)
            {
                if ((*idx)[i] == (*idx)[j])
                {
                    new_strides[new_nmode] += (*strides)[j];
                }
            }
            new_nmode++;
        }
    }
    new_idx = realloc(new_idx, new_nmode * sizeof(int64_t));
    new_extents = realloc(new_extents, new_nmode * sizeof(int64_t));
    new_strides = realloc(new_strides, new_nmode * sizeof(int64_t));
    free(*idx);
    free(*extents);
    free(*strides);
    *nmode = new_nmode;
    *idx = new_idx;
    *extents = new_extents;
    *strides = new_strides;
}

void* alloc_accum(HI_TAPP_prectype prec, HI_TAPP_datatype type)
{
    switch (prec)
    {
    case HI_TAPP_DEFAULT_PREC:
        switch (type)
        {
        case HI_TAPP_F32:
            return malloc(sizeof(float));
            break;
        case HI_TAPP_F64:
            return malloc(sizeof(double));
            break;
        case HI_TAPP_C32:
            return malloc(sizeof(complex float));
            break;
        case HI_TAPP_C64:
            return malloc(sizeof(complex double));
            break;
        case HI_TAPP_F16:
            return malloc(sizeof(_Float16));
            break;
        case HI_TAPP_BF16:
            //return malloc(sizeof(__bf16));
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F32F32_ACCUM_F32:
    case HI_TAPP_F16F16_ACCUM_F32:
    case HI_TAPP_BF16BF16_ACCUM_F32:
        switch (type)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            return malloc(sizeof(float));
            break;
        case HI_TAPP_C32:
        case HI_TAPP_C64:
            return malloc(sizeof(complex float));
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        switch (type)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            return malloc(sizeof(double));
            break;
        case HI_TAPP_C32:
        case HI_TAPP_C64:
            return malloc(sizeof(complex double));
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
        return malloc(sizeof(_Float16));
        break;
    default:
        break;
    }
    return NULL;
}

void* alloc_val(HI_TAPP_prectype prec, HI_TAPP_datatype type)
{
    switch (prec)
    {
    case HI_TAPP_DEFAULT_PREC:
        switch (type)
        {
        case HI_TAPP_F32:
            return malloc(sizeof(float));
            break;
        case HI_TAPP_F64:
            return malloc(sizeof(double));
            break;
        case HI_TAPP_C32:
            return malloc(sizeof(complex float));
            break;
        case HI_TAPP_C64:
            return malloc(sizeof(complex double));
            break;
        case HI_TAPP_F16:
            return malloc(sizeof(_Float16));
            break;
        case HI_TAPP_BF16:
            //return malloc(sizeof(__bf16));
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F32F32_ACCUM_F32:
        switch (type)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            return malloc(sizeof(float));
            break;
        case HI_TAPP_C32:
        case HI_TAPP_C64:
            return malloc(sizeof(complex float));
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        switch (type)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            return malloc(sizeof(double));
            break;
        case HI_TAPP_C32:
        case HI_TAPP_C64:
            return malloc(sizeof(complex double));
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
    case HI_TAPP_F16F16_ACCUM_F32:
        switch (type)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            return malloc(sizeof(_Float16));
            break;
        case HI_TAPP_C32:
        case HI_TAPP_C64:
            return malloc(sizeof(complex _Float16));
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_BF16BF16_ACCUM_F32:
        /*switch (type)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            return malloc(sizeof(__bf16));
            break;
        case HI_TAPP_C32:
        case HI_TAPP_C64:
            return malloc(sizeof(complex __bf16));
            break;
        default:
            break;
        }*/
        break;
    default:
        break;
    }
    return NULL;
}

void* create_prec_scalar(const void* scalar, HI_TAPP_datatype type, HI_TAPP_prectype prec)
{
    switch (type)
    {
    case HI_TAPP_F32:
        switch (prec)
        {
        case HI_TAPP_DEFAULT_PREC:
        case HI_TAPP_F32F32_ACCUM_F32:
        {
            float* prec_scalar = malloc(sizeof(float));
            *prec_scalar = *(float*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F64F64_ACCUM_F64:
        {
            double* prec_scalar = malloc(sizeof(double));
            *prec_scalar = *(float*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F16F16_ACCUM_F16:
        case HI_TAPP_F16F16_ACCUM_F32:
        {
            _Float16* prec_scalar = malloc(sizeof(_Float16));
            *prec_scalar = *(float*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_BF16BF16_ACCUM_F32:
        {
            /*__bf16* prec_alpha = malloc(sizeof(__bf16));
            *prec_scalar = *(float*)scalar;
            return prec_scalar;*/
            break;
        }
        default:
            break;
        }
    case HI_TAPP_F64:
        switch (prec)
        {
        case HI_TAPP_F32F32_ACCUM_F32:
        {
            float* prec_scalar = malloc(sizeof(float));
            *prec_scalar = *(double*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_DEFAULT_PREC:
        case HI_TAPP_F64F64_ACCUM_F64:
        {
            double* prec_scalar = malloc(sizeof(double));
            *prec_scalar = *(double*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F16F16_ACCUM_F16:
        case HI_TAPP_F16F16_ACCUM_F32:
        {
            _Float16* prec_scalar = malloc(sizeof(_Float16));
            *prec_scalar = *(double*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_BF16BF16_ACCUM_F32:
        {
            /*__bf16* prec_alpha = malloc(sizeof(__bf16));
            *prec_scalar = *(double*)scalar;
            return prec_scalar;*/
        }
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_C32:
        switch (prec)
        {
        case HI_TAPP_DEFAULT_PREC:
        case HI_TAPP_F32F32_ACCUM_F32:
        {
            complex float* prec_scalar = malloc(sizeof(complex float));
            *prec_scalar = *(complex float*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F64F64_ACCUM_F64:
        {
            complex double* prec_scalar = malloc(sizeof(complex double));
            *prec_scalar = *(complex float*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F16F16_ACCUM_F16:
        case HI_TAPP_F16F16_ACCUM_F32:
        {
            complex _Float16* prec_scalar = malloc(sizeof(complex _Float16));
            *prec_scalar = *(complex float*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_BF16BF16_ACCUM_F32:
        {
            /*complex __bf16* prec_alpha = malloc(sizeof(complex __bf16));
            *prec_scalar = *(complex float*)scalar;
            return prec_scalar;*/
            break;
        }
        default:
            break;
        }
        break;
    case HI_TAPP_C64:
        switch (prec)
        {
        case HI_TAPP_F32F32_ACCUM_F32:
        {
            complex float* prec_scalar = malloc(sizeof(complex float));
            *prec_scalar = *(complex double*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_DEFAULT_PREC:
        case HI_TAPP_F64F64_ACCUM_F64:
        {
            complex double* prec_scalar = malloc(sizeof(complex double));
            *prec_scalar = *(complex double*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F16F16_ACCUM_F16:
        case HI_TAPP_F16F16_ACCUM_F32:
        {
            complex _Float16* prec_scalar = malloc(sizeof(complex _Float16));
            *prec_scalar = *(complex double*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_BF16BF16_ACCUM_F32:
        {
            /*complex __bf16* prec_alpha = malloc(sizeof(complex __bf16));
            *prec_scalar = *(complex double*)scalar;
            return prec_scalar;*/
            break;
        }
        default:
            break;
        }
        break;
    case HI_TAPP_F16:
        switch (prec)
        {
        case HI_TAPP_F32F32_ACCUM_F32:
        {
            float* prec_scalar = malloc(sizeof(float));
            *prec_scalar = *(_Float16*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F64F64_ACCUM_F64:
        {
            double* prec_scalar = malloc(sizeof(double));
            *prec_scalar = *(_Float16*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_DEFAULT_PREC:
        case HI_TAPP_F16F16_ACCUM_F16:
        case HI_TAPP_F16F16_ACCUM_F32:
        {
            _Float16* prec_scalar = malloc(sizeof(_Float16));
            *prec_scalar = *(_Float16*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_BF16BF16_ACCUM_F32:
        {
            /*__bf16* prec_alpha = malloc(sizeof(__bf16));
            *prec_scalar = *(_Float16*)scalar;
            return prec_scalar;*/
            break;
        }
        default:
            break;
        }
        break;
    case HI_TAPP_BF16:
        switch (prec)
        {
        /*case HI_TAPP_F32F32_ACCUM_F32:
        {
            float* prec_scalar = malloc(sizeof(float));
            *prec_scalar = *(__bf16*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F64F64_ACCUM_F64:
        {
            double* prec_scalar = malloc(sizeof(double));
            *prec_scalar = *(__bf16*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_F16F16_ACCUM_F16:
        case HI_TAPP_F16F16_ACCUM_F32:
        {
            _Float16* prec_scalar = malloc(sizeof(_Float16));
            *prec_scalar = *(__bf16*)scalar;
            return prec_scalar;
            break;
        }
        case HI_TAPP_DEFAULT_PREC:
        case HI_TAPP_BF16BF16_ACCUM_F32:
        {
            __bf16* prec_alpha = malloc(sizeof(__bf16));
            *prec_scalar = *(__bf16*)scalar;
            return prec_scalar;
            break;
        }*/
        default:
            break;
        }
        break;
    default:
        return false;
        break;
    }
    return NULL;
}

bool is_complex(HI_TAPP_datatype type)
{
    switch (type)
    {
    case HI_TAPP_F32:
    case HI_TAPP_F64:
    case HI_TAPP_F16:
    case HI_TAPP_BF16:
        return false;
        break;
    case HI_TAPP_C32:
    case HI_TAPP_C64:
        return true;
        break;
    default:
        return false;
        break;
    }
}

void zero_sum(void* sum, HI_TAPP_prectype prec, HI_TAPP_datatype type, bool is_complex)
{
    switch (prec)
    {
    case HI_TAPP_DEFAULT_PREC:
        switch (type)
        {
        case HI_TAPP_F32:
            *((float*)sum) = 0;
            break;
        case HI_TAPP_F64:
            *((double*)sum) = 0;
            break;
        case HI_TAPP_C32:
            *((complex float*)sum) = 0;
            break;
        case HI_TAPP_C64:
            *((complex double*)sum) = 0;
            break;
        case HI_TAPP_F16:
            *((_Float16*)sum) = 0;
            break;
        case HI_TAPP_BF16:
            //*((__bf16*)sum) = 0;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F32F32_ACCUM_F32:
        if (is_complex)
        {
            *(complex float*)sum = 0;
        }
        else
        {
            *(float*)sum = 0;
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        if (is_complex)
        {
            *(complex double*)sum = 0;
        }
        else
        {
            *(double*)sum = 0;
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
    case HI_TAPP_F16F16_ACCUM_F32:
        if (is_complex)
        {
            *(complex _Float16*)sum = 0;
        }
        else
        {
            *(_Float16*)sum = 0;
        }
        break;
    case HI_TAPP_BF16BF16_ACCUM_F32:
        /*if (is_complex)
        {
            *(complex__bf16*)sum = 0;
        }
        else
        {
            *(__bf16*)sum = 0;
        }*/
        break;
    default:
        break;
    }
}

void zero_accum(void* accum, HI_TAPP_prectype prec, HI_TAPP_datatype type, bool is_complex)
{
    switch (prec)
    {
    case HI_TAPP_DEFAULT_PREC:
        switch (type)
        {
        case HI_TAPP_F32:
            *((float*)accum) = 0;
            break;
        case HI_TAPP_F64:
            *((double*)accum) = 0;
            break;
        case HI_TAPP_C32:
            *((complex float*)accum) = 0;
            break;
        case HI_TAPP_C64:
            *((complex double*)accum) = 0;
            break;
        case HI_TAPP_F16:
            *((_Float16*)accum) = 0;
            break;
        case HI_TAPP_BF16:
            //*((__bf16*)accum) = 0;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F32F32_ACCUM_F32:
    case HI_TAPP_F16F16_ACCUM_F32:
    case HI_TAPP_BF16BF16_ACCUM_F32:
        if (is_complex)
        {
            *(complex float*)accum = 0;
        }
        else
        {
            *(float*)accum = 0;
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        if (is_complex)
        {
            *(complex double*)accum = 0;
        }
        else
        {
            *(double*)accum = 0;
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
        if (is_complex)
        {
            *(complex _Float16*)accum = 0;
        }
        else
        {
            *(_Float16*)accum = 0;
        }
        break;
    default:
        break;
    }
}

bool is_equal(const void* val, HI_TAPP_datatype type, const void* comp_val, HI_TAPP_datatype comp_type)
{
    switch (type)
    {
    case HI_TAPP_F32:
        switch (comp_type)
        {
        case HI_TAPP_F32:
            return *(float*)val == *(float*)comp_val;
            break;
        case HI_TAPP_F64:
            return *(float*)val == *(double*)comp_val;
            break;
        case HI_TAPP_C32:
            return *(float*)val == *(complex float*)comp_val;
            break;
        case HI_TAPP_C64:
            return *(float*)val == *(complex double*)comp_val;
            break;
        case HI_TAPP_F16:
            return *(float*)val == *(_Float16*)comp_val;
            break;
        case HI_TAPP_BF16:
            //return *(float*)val == *(__bf16*)comp_val;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F64:
        switch (comp_type)
        {
        case HI_TAPP_F32:
            return *(double*)val == *(float*)comp_val;
            break;
        case HI_TAPP_F64:
            return *(double*)val == *(double*)comp_val;
            break;
        case HI_TAPP_C32:
            return *(double*)val == *(complex float*)comp_val;
            break;
        case HI_TAPP_C64:
            return *(double*)val == *(complex double*)comp_val;
            break;
        case HI_TAPP_F16:
            return *(double*)val == *(_Float16*)comp_val;
            break;
        case HI_TAPP_BF16:
            //return *(double*)val == *(__bf16*)comp_val;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_C32:
        switch (comp_type)
        {
        case HI_TAPP_F32:
            return *(complex float*)val == *(float*)comp_val;
            break;
        case HI_TAPP_F64:
            return *(complex float*)val == *(double*)comp_val;
            break;
        case HI_TAPP_C32:
            return *(complex float*)val == *(complex float*)comp_val;
            break;
        case HI_TAPP_C64:
            return *(complex float*)val == *(complex double*)comp_val;
            break;
        case HI_TAPP_F16:
            return *(complex float*)val == *(_Float16*)comp_val;
            break;
        case HI_TAPP_BF16:
            //return *(complex float*)val == *(__bf16*)comp_val;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_C64:
        switch (comp_type)
        {
        case HI_TAPP_F32:
            return *(complex double*)val == *(float*)comp_val;
            break;
        case HI_TAPP_F64:
            return *(complex double*)val == *(double*)comp_val;
            break;
        case HI_TAPP_C32:
            return *(complex double*)val == *(complex float*)comp_val;
            break;
        case HI_TAPP_C64:
            return *(complex double*)val == *(complex double*)comp_val;
            break;
        case HI_TAPP_F16:
            return *(complex double*)val == *(_Float16*)comp_val;
            break;
        case HI_TAPP_BF16:
            //return *(complex double*)val == *(__bf16*)comp_val;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F16:
        switch (comp_type)
        {
        case HI_TAPP_F32:
            return *(_Float16*)val == *(float*)comp_val;
            break;
        case HI_TAPP_F64:
            return *(_Float16*)val == *(double*)comp_val;
            break;
        case HI_TAPP_C32:
            return *(_Float16*)val == *(complex float*)comp_val;
            break;
        case HI_TAPP_C64:
            return *(_Float16*)val == *(complex double*)comp_val;
            break;
        case HI_TAPP_F16:
            return *(_Float16*)val == *(_Float16*)comp_val;
            break;
        case HI_TAPP_BF16:
            //return *(_Float16*)val == *(__bf16*)comp_val;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_BF16:
        /*switch (comp_type)
        {
        case HI_TAPP_F32:
            return *(__bf16*)val == *(float*)comp_val;
            break;
        case HI_TAPP_F64:
            return *(__bf16*)val == *(double*)comp_val;
            break;
        case HI_TAPP_C32:
            return *(__bf16*)val == *(complex float*)comp_val;
            break;
        case HI_TAPP_C64:
            return *(__bf16*)val == *(complex double*)comp_val;
            break;
        case HI_TAPP_F16:
            return *(__bf16*)val == *(_Float16*)comp_val;
            break;
        case HI_TAPP_BF16:
            //return *(__bf16*)val == *(__bf16*)comp_val;
            break;
        default:
            break;
        }*/
        break;
    default:
        break;
    }
    return false;
}

void sum_unary_contractions(void* sum, const void* tensor, int index, HI_TAPP_element_op op, HI_TAPP_datatype type, HI_TAPP_prectype prec)
{
    switch (prec)
    {
    case HI_TAPP_DEFAULT_PREC:
        switch (type)
        {
        case HI_TAPP_F32:
            *(float*)sum += ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(double*)sum += ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex float*)sum += op == HI_TAPP_CONJUGATE ? conjf(((complex float*)tensor)[index]) : ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex double*)sum += op == HI_TAPP_CONJUGATE ? conj(((complex double*)tensor)[index]) : ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(_Float16*)sum += ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            //*(__bf16*)sum += ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F32F32_ACCUM_F32:
        switch (type)
        {
        case HI_TAPP_F32:
            *(float*)sum += ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(float*)sum += ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex float*)sum += op == HI_TAPP_CONJUGATE ? conjf(((complex float*)tensor)[index]) : ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex float*)sum += op == HI_TAPP_CONJUGATE ? conj(((complex double*)tensor)[index]) : ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(float*)sum += ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            //*(float*)sum += ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        switch (type)
        {
        case HI_TAPP_F32:
            *(double*)sum += ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(double*)sum += ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex double*)sum += op == HI_TAPP_CONJUGATE ? conjf(((complex float*)tensor)[index]) : ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex double*)sum += op == HI_TAPP_CONJUGATE ? conj(((complex double*)tensor)[index]) : ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(double*)sum += ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            //*(double*)sum += ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
    case HI_TAPP_F16F16_ACCUM_F32:
        switch (type)
        {
        case HI_TAPP_F32:
            *(_Float16*)sum += ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(_Float16*)sum += ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex _Float16*)sum += ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex _Float16*)sum += ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(_Float16*)sum += ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            //*(_Float16*)sum += ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_BF16BF16_ACCUM_F32:
        /*switch (type)
        {
        case HI_TAPP_F32:
            *(__bf16*)sum += ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(__bf16*)sum += ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex __bf16*)sum += ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex __bf16*)sum += ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(__bf16*)sum += ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            *(__bf16*)sum += ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }*/
        break;
    
    default:
        break;
    }
}

void calculate_beta_C(const void* beta, HI_TAPP_datatype type_beta, bool is_complex_beta, const void* val_C, HI_TAPP_datatype type_C, bool is_complex_C, HI_TAPP_element_op op_C, HI_TAPP_prectype prec, void* accum, HI_TAPP_datatype type_accum, bool is_complex_accum)
{
    if (prec == HI_TAPP_DEFAULT_PREC)
    {
        calculate_beta_C_default(beta, type_beta, val_C, type_C, op_C, accum, type_accum);
    }
    else
    {
        calculate_beta_C_prec(beta, is_complex_beta, val_C, is_complex_C, prec, accum, is_complex_accum);
    }
}

void calculate_beta_C_default(const void* beta, HI_TAPP_datatype type_beta, const void* val_C, HI_TAPP_datatype type_C, HI_TAPP_element_op op_C, void* accum, HI_TAPP_datatype type_accum)
{
    switch (type_accum)
    {
    case HI_TAPP_F32:
        switch (type_C)
        {
        case HI_TAPP_F32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(float*)accum = *(float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F64:
                *(float*)accum = *(double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(float*)accum = *(__bf16*)beta * *(float*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(float*)accum = *(float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F64:
                *(float*)accum = *(double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(float*)accum = *(__bf16*)beta * *(double*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(float*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F64:
                *(float*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C32:
                *(float*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C64:
                *(float*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F16:
                *(float*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(float*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(float*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F64:
                *(float*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C32:
                *(float*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C64:
                *(float*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F16:
                *(float*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(float*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(float*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F64:
                *(float*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(float*)accum = *(__bf16*)beta * *(_Float16*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_beta)
            {
            case HI_TAPP_F32:
                *(float*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F64:
                *(float*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(__bf16*)val_C;
                break;
            default:
                break;
            }*/
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F64:
        switch (type_C)
        {
        case HI_TAPP_F32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(double*)accum = *(float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F64:
                *(double*)accum = *(double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(double*)accum = *(__bf16*)beta * *(float*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(double*)accum = *(float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F64:
                *(double*)accum = *(double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(double*)accum = *(__bf16*)beta * *(double*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(double*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F64:
                *(double*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C32:
                *(double*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C64:
                *(double*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F16:
                *(double*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(double*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(double*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F64:
                *(double*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C32:
                *(double*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C64:
                *(double*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F16:
                *(double*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(double*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(double*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F64:
                *(double*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(double*)accum = *(__bf16*)beta * *(_Float16*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_beta)
            {
            case HI_TAPP_F32:
                *(double*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F64:
                *(double*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(__bf16*)val_C;
                break;
            default:
                break;
            }*/
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_C32:
        switch (type_C)
        {
        case HI_TAPP_F32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex float*)accum = *(float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(complex float*)accum = *(__bf16*)beta * *(float*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex float*)accum = *(float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(complex float*)accum = *(__bf16*)beta * *(double*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex float*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F64:
                *(complex float*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(complex float*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex float*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F64:
                *(complex float*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(complex float*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex float*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(complex float*)accum = *(__bf16*)beta * *(_Float16*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex float*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(__bf16*)val_C;
                break;
            default:
                break;
            }*/
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_C64:
        switch (type_C)
        {
        case HI_TAPP_F32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex double*)accum = *(float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(complex double*)accum = *(__bf16*)beta * *(float*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex double*)accum = *(float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(complex double*)accum = *(__bf16*)beta * *(double*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex double*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F64:
                *(complex double*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(complex double*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex double*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F64:
                *(complex double*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(complex double*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex double*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(complex double*)accum = *(__bf16*)beta * *(_Float16*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_beta)
            {
            case HI_TAPP_F32:
                *(complex double*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(__bf16*)val_C;
                break;
            default:
                break;
            }*/
            break;
        default:
            break;
        }
    case HI_TAPP_F16:
        switch (type_C)
        {
        case HI_TAPP_F32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(_Float16*)accum = *(float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(_Float16*)accum = *(__bf16*)beta * *(float*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(_Float16*)accum = *(float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(_Float16*)accum = *(__bf16*)beta * *(double*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(_Float16*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F64:
                *(_Float16*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(_Float16*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(_Float16*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F64:
                *(_Float16*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_BF16:
                //*(_Float16*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(_Float16*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_BF16:
                //*(_Float16*)accum = *(__bf16*)beta * *(_Float16*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_beta)
            {
            case HI_TAPP_F32:
                *(_Float16*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(__bf16*)val_C;
                break;
            default:
                break;
            }*/
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_BF16:
        /*switch (type_C)
        {
        case HI_TAPP_F32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(__bf16*)accum = *(float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case HI_TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(float*)val_C;
                break;
            case HI_TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
            case HI_TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(float*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(__bf16*)accum = *(float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case HI_TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(double*)val_C;
                break;
            case HI_TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
            case HI_TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(double*)val_C;
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(__bf16*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F64:
                *(__bf16*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case HI_TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(__bf16*)accum = *(float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F64:
                *(__bf16*)accum = *(double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case HI_TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * (op_C == HI_TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(__bf16*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
            case HI_TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(_Float16*)val_C);
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            switch (type_beta)
            {
            case HI_TAPP_F32:
                *(__bf16*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
            case HI_TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(__bf16*)val_C;
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

void calculate_beta_C_prec(const void* beta, bool is_complex_beta, const void* val_C, bool is_complex_C, HI_TAPP_prectype prec, void* accum, bool is_complex_accum)
{
    switch (prec)
    {
    case HI_TAPP_F32F32_ACCUM_F32:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum += *(complex float*)beta * *(complex float*)val_C;
                }
                else
                {
                    *(complex float*)accum += *(float*)beta * *(complex float*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum += *(complex float*)beta * *(float*)val_C;
                }
                else
                {
                    *(complex float*)accum += *(float*)beta * *(float*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum += *(complex float*)beta * *(complex float*)val_C;
                }
                else
                {
                    *(float*)accum += *(float*)beta * *(complex float*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum += *(complex float*)beta * *(float*)val_C;
                }
                else
                {
                    *(float*)accum += *(float*)beta * *(float*)val_C;
                }
            }
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex double*)accum += *(complex double*)beta * *(complex double*)val_C;
                }
                else
                {
                    *(complex double*)accum += *(double*)beta * *(complex double*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex double*)accum += *(complex double*)beta * *(double*)val_C;
                }
                else
                {
                    *(complex double*)accum += *(double*)beta * *(double*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(double*)accum += *(complex double*)beta * *(complex double*)val_C;
                }
                else
                {
                    *(double*)accum += *(double*)beta * *(complex double*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(double*)accum += *(complex double*)beta * *(double*)val_C;
                }
                else
                {
                    *(double*)accum += *(double*)beta * *(double*)val_C;
                }
            }
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex _Float16*)accum += *(complex _Float16*)beta * *(complex _Float16*)val_C;
                }
                else
                {
                    *(complex _Float16*)accum += *(_Float16*)beta * *(complex _Float16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex _Float16*)accum += *(complex _Float16*)beta * *(_Float16*)val_C;
                }
                else
                {
                    *(complex _Float16*)accum += *(_Float16*)beta * *(_Float16*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(_Float16*)accum += *(complex _Float16*)beta * *(complex _Float16*)val_C;
                }
                else
                {
                    *(_Float16*)accum += *(_Float16*)beta * *(complex _Float16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(_Float16*)accum += *(complex _Float16*)beta * *(_Float16*)val_C;
                }
                else
                {
                    *(_Float16*)accum += *(_Float16*)beta * *(_Float16*)val_C;
                }
            }
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F32:
        if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum += *(complex _Float16*)beta * *(complex _Float16*)val_C;
                }
                else
                {
                    *(complex float*)accum += *(_Float16*)beta * *(complex _Float16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum += *(complex _Float16*)beta * *(_Float16*)val_C;
                }
                else
                {
                    *(complex float*)accum += *(_Float16*)beta * *(_Float16*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum += *(complex _Float16*)beta * *(complex _Float16*)val_C;
                }
                else
                {
                    *(float*)accum += *(_Float16*)beta * *(complex _Float16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum += *(complex _Float16*)beta * *(_Float16*)val_C;
                }
                else
                {
                    *(float*)accum += *(_Float16*)beta * *(_Float16*)val_C;
                }
            }
        }
        break;
    case HI_TAPP_BF16BF16_ACCUM_F32:
        /*if (is_complex_accum)
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum += *(complex __bf16*)beta * *(complex __bf16*)val_C;
                }
                else
                {
                    *(complex float*)accum += *(__bf16*)beta * *(complex __bf16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum += *(complex __bf16*)beta * *(__bf16*)val_C;
                }
                else
                {
                    *(complex float*)accum += *(__bf16*)beta * *(__bf16*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum += *(complex __bf16*)beta * *(complex __bf16*)val_C;
                }
                else
                {
                    *(float*)accum += *(__bf16*)beta * *(complex __bf16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum += *(complex __bf16*)beta * *(__bf16*)val_C;
                }
                else
                {
                    *(float*)accum += *(__bf16*)beta * *(__bf16*)val_C;
                }
            }
        }*/
        break;
    default:
        break;
    }
}

void calculate_alpha_A_B(const void* alpha, HI_TAPP_datatype type_alpha, bool is_complex_alpha, const void* sum_A, HI_TAPP_datatype type_A, bool is_complex_A, const void* sum_B, HI_TAPP_datatype type_B, bool is_complex_B, HI_TAPP_prectype prec, void* accum, HI_TAPP_datatype type_accum, bool is_complex_accum)
{
    if (prec == HI_TAPP_DEFAULT_PREC)
    {
        calculate_alpha_A_B_default(alpha, type_alpha, sum_A, type_A, sum_B, type_B, accum, type_accum);
    }
    else 
    {
        calculate_alpha_A_B_prec(alpha, is_complex_alpha, sum_A, is_complex_A, sum_B, is_complex_B, prec, accum, is_complex_accum);
    }
}

void calculate_alpha_A_B_default(const void* alpha, HI_TAPP_datatype type_alpha, const void* sum_A, HI_TAPP_datatype type_A, const void* sum_B, HI_TAPP_datatype type_B, void* accum, HI_TAPP_datatype type_accum)
{
    switch (type_accum)
    {
    case HI_TAPP_F32:
        switch (type_A)
        {
        case HI_TAPP_F32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
    case HI_TAPP_F64:
        switch (type_A)
        {
        case HI_TAPP_F32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
    case HI_TAPP_C32:
        switch (type_A)
        {
        case HI_TAPP_F32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex float*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex float*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex float*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex float*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex float*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex float*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
    case HI_TAPP_C64:
        switch (type_A)
        {
        case HI_TAPP_F32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(complex double*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(complex double*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(complex double*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(complex double*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(complex double*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(complex double*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
    case HI_TAPP_F16:
        switch (type_A)
        {
        case HI_TAPP_F32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    //*(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                /*switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }*/
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            /*switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(_Float16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(_Float16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(_Float16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(_Float16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(_Float16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(_Float16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
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
        break;
    case HI_TAPP_BF16:
        /*switch (type_A)
        {
        case HI_TAPP_F32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C32:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex float*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_C64:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(complex double*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_F16:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(_Float16*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
            break;
        case HI_TAPP_BF16:
            switch (type_B)
            {
            case HI_TAPP_F32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(double*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C32:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex float*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_C64:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(complex double*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(complex doubles*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_F16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(_Float16*)sum_B;
                    break;
                default:
                    break;
                }
                break;
            case HI_TAPP_BF16:
                switch (type_alpha)
                {
                case HI_TAPP_F32:
                    *(__bf16*)accum += *(float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F64:
                    *(__bf16*)accum += *(double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C32:
                    *(__bf16*)accum += *(complex float*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_C64:
                    *(__bf16*)accum += *(complex double*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_F16:
                    *(__bf16*)accum += *(_Float16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                case HI_TAPP_BF16:
                    *(__bf16*)accum += *(__bf16*)alpha * *(__bf16*)sum_A * *(__bf16*)sum_B;
                    break;
                default:
                    break;
                }
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

void calculate_alpha_A_B_prec(const void* alpha, bool is_complex_alpha, const void* sum_A, bool is_complex_A, const void* sum_B, bool is_complex_B, HI_TAPP_prectype prec, void* accum, bool is_complex_accum)
{
    switch (prec)
    {
    case HI_TAPP_F32F32_ACCUM_F32:
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
    case HI_TAPP_F64F64_ACCUM_F64:
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
    case HI_TAPP_F16F16_ACCUM_F16:
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
    case HI_TAPP_F16F16_ACCUM_F32:
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
    case HI_TAPP_BF16BF16_ACCUM_F32:
        /*if (is_complex_accum)
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
        }*/
        break;
    default:
        break;
    }
}

void calculate_op_D(void* accum, HI_TAPP_datatype type_D, HI_TAPP_element_op op_D, HI_TAPP_prectype prec)
{
    switch (prec)
    {
    case HI_TAPP_DEFAULT_PREC:
        switch (type_D)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
            break;
        case HI_TAPP_C32:
            if (op_D == HI_TAPP_CONJUGATE)
            {
                *(complex float*)accum = conjf(*(complex float*)accum);
            }
            break;
        case HI_TAPP_C64:
            if (op_D == HI_TAPP_CONJUGATE)
            {
                *(complex double*)accum = conj(*(complex double*)accum);
            }
            break;
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F32F32_ACCUM_F32:
    case HI_TAPP_F16F16_ACCUM_F32:
    case HI_TAPP_BF16BF16_ACCUM_F32:
        switch (type_D)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            break;
        case HI_TAPP_C32:
        case HI_TAPP_C64:
            if (op_D == HI_TAPP_CONJUGATE)
            {
                *(complex float*)accum = conjf(*(complex float*)accum);
            }
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        switch (type_D)
        {
        case HI_TAPP_F32:
        case HI_TAPP_F64:
        case HI_TAPP_F16:
        case HI_TAPP_BF16:
            break;
        case HI_TAPP_C32:
        case HI_TAPP_C64:
            if (op_D == HI_TAPP_CONJUGATE)
            {
                *(complex double*)accum = conjf(*(complex double*)accum);
            }
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
        break;
    default:
        break;
    }
}

void get_val(void* val, const void* tensor, int64_t index, HI_TAPP_datatype type, HI_TAPP_prectype prec)
{
    switch (prec)
    {
    case HI_TAPP_DEFAULT_PREC:
        switch (type)
        {
        case HI_TAPP_F32:
            *(float*)val = ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(double*)val = ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex float*)val = ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex double*)val = ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(_Float16*)val = ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            //*(__bf16*)val = ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F32F32_ACCUM_F32:
        switch (type)
        {
        case HI_TAPP_F32:
            *(float*)val = ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(float*)val = ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex float*)val = ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex float*)val = ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(float*)val = ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            //*(float*)val = ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        switch (type)
        {
        case HI_TAPP_F32:
            *(double*)val = ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(double*)val = ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex double*)val = ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex double*)val = ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(double*)val = ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            //*(double*)val = ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
    case HI_TAPP_F16F16_ACCUM_F32:
        switch (type)
        {
        case HI_TAPP_F32:
            *(_Float16*)val = ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(_Float16*)val = ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex _Float16*)val = ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex _Float16*)val = ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(_Float16*)val = ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            //*(_Float16*)val = ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_BF16BF16_ACCUM_F32:
        /*switch (type)
        {
        case HI_TAPP_F32:
            *(__bf16*)val = ((float*)tensor)[index];
            break;
        case HI_TAPP_F64:
            *(__bf16*)val = ((double*)tensor)[index];
            break;
        case HI_TAPP_C32:
            *(complex __bf16*)val = ((complex float*)tensor)[index];
            break;
        case HI_TAPP_C64:
            *(complex __bf16*)val = ((complex double*)tensor)[index];
            break;
        case HI_TAPP_F16:
            *(__bf16*)val = ((_Float16*)tensor)[index];
            break;
        case HI_TAPP_BF16:
            *(__bf16*)val = ((__bf16*)tensor)[index];
            break;
        default:
            break;
        }*/
        break;
    
    default:
        break;
    }
}

void assign_D(void* D, HI_TAPP_datatype type_D, int64_t index_D, void* accum, HI_TAPP_prectype prec)
{
    switch (prec)
    {
    case HI_TAPP_DEFAULT_PREC:
        switch (type_D)
        {
        case HI_TAPP_F32:
            ((float*)D)[index_D] = *(float*)accum;
            break;
        case HI_TAPP_F64:
            ((double*)D)[index_D] = *(double*)accum;
            break;
        case HI_TAPP_C32:
            ((complex float*)D)[index_D] = *(complex float*)accum;
            break;
        case HI_TAPP_C64:
            ((complex double*)D)[index_D] = *(complex double*)accum;
            break;
        case HI_TAPP_F16:
            ((_Float16*)D)[index_D] = *(_Float16*)accum;
            break;
        case HI_TAPP_BF16:
            //((__bf16*)D)[index_D] = *(__bf16*)accum;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F32F32_ACCUM_F32:
    case HI_TAPP_F16F16_ACCUM_F32:
    case HI_TAPP_BF16BF16_ACCUM_F32:
        switch (type_D)
        {
        case HI_TAPP_F32:
            ((float*)D)[index_D] = *(float*)accum;
            break;
        case HI_TAPP_F64:
            ((double*)D)[index_D] = *(float*)accum;
            break;
        case HI_TAPP_C32:
            ((complex float*)D)[index_D] = *(complex float*)accum;
            break;
        case HI_TAPP_C64:
            ((complex double*)D)[index_D] = *(complex float*)accum;
            break;
        case HI_TAPP_F16:
            ((_Float16*)D)[index_D] = *(float*)accum;
            break;
        case HI_TAPP_BF16:
            //((__bf16*)D)[index_D] = *(float*)accum;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F64F64_ACCUM_F64:
        switch (type_D)
        {
        case HI_TAPP_F32:
            ((float*)D)[index_D] = *(double*)accum;
            break;
        case HI_TAPP_F64:
            ((double*)D)[index_D] = *(double*)accum;
            break;
        case HI_TAPP_C32:
            ((complex float*)D)[index_D] = *(complex double*)accum;
            break;
        case HI_TAPP_C64:
            ((complex double*)D)[index_D] = *(complex double*)accum;
            break;
        case HI_TAPP_F16:
            ((_Float16*)D)[index_D] = *(double*)accum;
            break;
        case HI_TAPP_BF16:
            //((__bf16*)D)[index_D] = *(double*)accum;
            break;
        default:
            break;
        }
        break;
    case HI_TAPP_F16F16_ACCUM_F16:
        switch (type_D)
        {
        case HI_TAPP_F32:
            ((float*)D)[index_D] = *(_Float16*)accum;
            break;
        case HI_TAPP_F64:
            ((double*)D)[index_D] = *(_Float16*)accum;
            break;
        case HI_TAPP_C32:
            break;
        case HI_TAPP_C64:
            break;
        case HI_TAPP_F16:
            ((_Float16*)D)[index_D] = *(_Float16*)accum;
            break;
        case HI_TAPP_BF16:
            //((__bf16*)D)[index_D] = *(_Float16*)accum;
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}
