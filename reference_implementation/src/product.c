/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */
#include "../include/product.h"

int extract_binary_contractions_indices(int nmode_A, int nmode_B, int nmode_D, const int64_t* idx_A, const int64_t* idx_B, const int64_t* idx_D, int64_t** idx_contraction_ptr);
int extract_unary_contracted_indices(int nmode, int64_t* idx, int nmode_1, int64_t* idx_1, int nmode_2, int64_t* idx_2, int64_t** idx_unary_contractions_ptr);
void extract_extents(int nr_extents, int64_t* idx_extraction, int nmode, const int64_t* idx, int64_t* extents, int64_t** extracted_extents_ptr);
void compile_strides(int64_t* strides, int ndim, const int64_t* idx, int ndim_D, const int64_t* idx_D, int contractions, int64_t* idx_contraction, int64_t* free_strides, int64_t* contracted_strides);
int64_t calculate_size(int64_t* extents, int nmode);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
void zero_array(int64_t* arr, int size);
void extract_free_strides(int nmode, const int64_t* idx, int64_t* strides, int nmode_D, const int64_t* idx_D, int64_t** strides_free_ptr);
void extract_contracted_strides(int nmode, const int64_t* idx, int64_t* strides, int contractions, int64_t* idx_contraction, int64_t** strides_contractions_ptr);
void sum_unary_contractions(void* sum, const void* tensor, int index, TAPP_element_op op, TAPP_datatype type, TAPP_prectype prec);
void calculate_beta_C(const void* beta, TAPP_datatype type_beta, bool is_complex_beta, const void* val_C, TAPP_datatype type_C, bool is_complex_C, TAPP_element_op op_C, TAPP_prectype prec, void* accum, TAPP_datatype type_accum, bool is_complex_accum);
void calculate_beta_C_default(const void* beta, TAPP_datatype type_beta, const void* val_C, TAPP_datatype type_C, TAPP_element_op op_C, void* accum, TAPP_datatype type_accum);
void calculate_beta_C_prec(const void* beta, bool is_complex_beta, const void* val_C, bool is_complex_C, TAPP_prectype prec, void* accum, bool is_complex_accum);
void calculate_alpha_A_B(const void* alpha, TAPP_datatype type_alpha, bool is_complex_alpha, const void* sum_A, TAPP_datatype type_A, bool is_complex_A, const void* sum_B, TAPP_datatype type_B, bool is_complex_B, TAPP_prectype prec, void* accum, TAPP_datatype type_accum, bool is_complex_accum);
void calculate_alpha_A_B_default(const void* alpha, TAPP_datatype type_alpha, const void* sum_A, TAPP_datatype type_A, const void* sum_B, TAPP_datatype type_B, void* accum, TAPP_datatype type_accum);
void calculate_alpha_A_B_prec(const void* alpha, bool is_complex_alpha, const void* sum_A, bool is_complex_A, const void* sum_B, bool is_complex_B, TAPP_prectype prec, void* accum, bool is_complex_accum);
void calculate_op_D(void* accum, TAPP_datatype type_D, TAPP_element_op op_D, TAPP_prectype prec);
void get_val(void* val, const void* tensor, int64_t index, TAPP_datatype type, TAPP_prectype prec);
void assign_D(void* D, TAPP_datatype type_D, int64_t index_D, void* accum, TAPP_prectype prec);
int check_repeated_idx(int nmode, const int64_t* idx, int error_code);
int check_idx_occurrence(int nmode_origin, const int64_t* idx_origin, int nmode_test_A, const int64_t* idx_test_A, int nmode_test_B, const int64_t* idx_test_B, int unique_idx_code);
int check_extents(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_D, const int64_t* idx_D, const int64_t* extents_D, int missmatch_AA_code, int missmatch_AB_code, int missmatch_AD_code);
int check_same_structure(int nmode_A, const int64_t* idx_A, const int64_t* extents_A, int nmode_B, const int64_t* idx_B, const int64_t* extents_B, int nmode_code, int idx_code, int extent_code);
int check_self_aliasing(int nmode, const int64_t* extents, const int64_t* strides, int error_code);
int check_tensor_existence(const void* scalar, TAPP_datatype type, const void* tensor, int error_code);
int check_executor_existence(TAPP_executor exec, int error_code);
void merge_sort_strides(int64_t* strides, int64_t*extents, int left, int right);
void merge_strides(int64_t* strides, int64_t* extents, int left, int mid, int right);
void* alloc_accum(TAPP_prectype prec, TAPP_datatype type);
void* alloc_val(TAPP_prectype prec, TAPP_datatype type);
void* create_prec_scalar(const void* scalar, TAPP_datatype type, TAPP_prectype prec);
void* alloc_alpha(TAPP_prectype prec, TAPP_datatype type);
void* alloc_beta(TAPP_prectype prec, TAPP_datatype type);
bool is_complex(TAPP_datatype type);
void zero_sum(void* sum, TAPP_prectype prec, TAPP_datatype type, bool is_complex);
void zero_accum(void* accum, TAPP_prectype prec, TAPP_datatype type, bool is_complex);
bool is_equal(const void* val, TAPP_datatype type, const void* comp_val, TAPP_datatype comp_type);
void compress_repeated_indices(int* nmode, int64_t** idx, int64_t** extents, int64_t** strides);
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

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
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
                                void* D)
{
    struct plan* plan_ptr = (struct plan*)plan;
    TAPP_handle handle = plan_ptr->handle;

    TAPP_element_op op_A = plan_ptr->op_A;
    TAPP_tensor_info A_info = (TAPP_tensor_info)(plan_ptr->A);
    struct tensor_info* A_info_ptr = (struct tensor_info*)(plan_ptr->A);

    TAPP_element_op op_B = plan_ptr->op_B;
    TAPP_tensor_info B_info = (TAPP_tensor_info)(plan_ptr->B);
    struct tensor_info* B_info_ptr = (struct tensor_info*)(plan_ptr->B);

    TAPP_element_op op_C = plan_ptr->op_C;
    TAPP_tensor_info C_info = (TAPP_tensor_info)(plan_ptr->C);
    struct tensor_info* C_info_ptr = (struct tensor_info*)(plan_ptr->C);

    TAPP_element_op op_D = plan_ptr->op_D;
    TAPP_tensor_info D_info = (TAPP_tensor_info)(plan_ptr->D);
    struct tensor_info* D_info_ptr = (struct tensor_info*)(plan_ptr->D);

    TAPP_prectype prec = plan_ptr->prec;

    TAPP_datatype type_A = A_info_ptr->type;
    int nmode_A = TAPP_get_nmodes(A_info);
    int64_t* extents_A = malloc(nmode_A * sizeof(int64_t));
    TAPP_get_extents(A_info, extents_A);
    int64_t* strides_A = malloc(nmode_A * sizeof(int64_t));
    TAPP_get_strides(A_info, strides_A);
    int64_t* idx_A = malloc(nmode_A * sizeof(int64_t));
    memcpy(idx_A, plan_ptr->idx_A, nmode_A * sizeof(int64_t));

    TAPP_datatype type_B = B_info_ptr->type;
    int nmode_B = TAPP_get_nmodes(B_info);
    int64_t* extents_B = malloc(nmode_B * sizeof(int64_t));
    TAPP_get_extents(B_info, extents_B);
    int64_t* strides_B = malloc(nmode_B * sizeof(int64_t));
    TAPP_get_strides(B_info, strides_B);
    int64_t* idx_B = malloc(nmode_B * sizeof(int64_t));
    memcpy(idx_B, plan_ptr->idx_B, nmode_B * sizeof(int64_t));

    TAPP_datatype type_C = C_info_ptr->type;
    int nmode_C = TAPP_get_nmodes(C_info);
    int64_t* extents_C = malloc(nmode_C * sizeof(int64_t));
    TAPP_get_extents(C_info, extents_C);
    int64_t* strides_C = malloc(nmode_C * sizeof(int64_t));
    TAPP_get_strides(C_info, strides_C);
    int64_t* idx_C = malloc(nmode_C * sizeof(int64_t));
    memcpy(idx_C, plan_ptr->idx_C, nmode_C * sizeof(int64_t));

    TAPP_datatype type_D = D_info_ptr->type;
    int nmode_D = TAPP_get_nmodes(D_info);
    int64_t* extents_D = malloc(nmode_D * sizeof(int64_t));
    TAPP_get_extents(D_info, extents_D);
    int64_t* strides_D = malloc(nmode_D * sizeof(int64_t));
    TAPP_get_strides(D_info, strides_D);
    int64_t* idx_D = malloc(nmode_D * sizeof(int64_t));
    memcpy(idx_D, plan_ptr->idx_D, nmode_D * sizeof(int64_t));

    int error_status = 0;

    if (error_status == 0) error_status = check_idx_occurrence(nmode_D, idx_D, nmode_A, idx_A, nmode_B, idx_B, 4);
    if (error_status == 0) error_status = check_extents(nmode_A, idx_A, extents_A, nmode_B, idx_B, extents_B, nmode_D, idx_D, extents_D, 9, 1, 2);
    if (error_status == 0) error_status = check_extents(nmode_B, idx_B, extents_B, nmode_A, idx_A, extents_A, nmode_D, idx_D, extents_D, 10, 1, 3);
    if (error_status == 0) error_status = check_extents(nmode_D, idx_D, extents_D, nmode_A, idx_A, extents_A, nmode_B, idx_B, extents_B, 11, 2, 3);
    if (error_status == 0) error_status = check_same_structure(nmode_C, idx_C, extents_C, nmode_D, idx_D, extents_D, 5, 6, 7);
    if (error_status == 0) error_status = check_self_aliasing(nmode_D, extents_D, strides_D, 8);
    if (error_status == 0) error_status = check_tensor_existence(beta, type_D, C, 12);
    if (error_status == 0) error_status = check_executor_existence(exec, 33);
    if (error_status != 0)
    {
        free(idx_A);
        free(extents_A);
        free(strides_A);
        free(idx_B);
        free(extents_B);
        free(strides_B);
        free(idx_C);
        free(extents_C);
        free(strides_C);
        free(idx_D);
        free(extents_D);
        free(strides_D);
        return error_status;
    }
    int64_t size_D;

    intptr_t* exec_ptr= &exec; //pointer to intptr_t (TAPP_executor)
    int* exec_int_ptr = (int*) *exec_ptr;//dereference to get the int pointer

    void* E_ = D;
    if((*exec_int_ptr) == 12 ) { // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
      size_D = calculate_size(extents_D, nmode_D);
      int64_t in_bytes;
      switch (type_D) { // tapp_datatype
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
      bind_tblis_execute_product(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                       nmode_B, extents_B, strides_B, B, op_B, idx_B,
                       nmode_C, extents_C, strides_C, C, op_C, idx_D,
                       nmode_D, extents_D, strides_D, E_, op_D, idx_D,
                       alpha, beta, type_D);
#endif
    }

    if((*exec_int_ptr) == 1 || (*exec_int_ptr) == 12 ) { // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
      // if((*exec_int_ptr) == 1) printf("tapp used1 \n");
      compress_repeated_indices(&nmode_A, &idx_A, &extents_A, &strides_A);
      compress_repeated_indices(&nmode_B, &idx_B, &extents_B, &strides_B);
      compress_repeated_indices(&nmode_C, &idx_C, &extents_C, &strides_C);
      compress_repeated_indices(&nmode_D, &idx_D, &extents_D, &strides_D);

      int64_t* idx_binary_contractions = NULL;
      int binary_contractions = extract_binary_contractions_indices(nmode_A, nmode_B, nmode_D, idx_A, idx_B, idx_D, &idx_binary_contractions);

      int64_t* extents_binary_contractions = NULL;
      extract_extents(binary_contractions, idx_binary_contractions, nmode_A, idx_A, extents_A, &extents_binary_contractions);

      int size_binary_contractions = calculate_size(extents_binary_contractions, binary_contractions);

      int64_t* idx_unary_contractions_A = NULL;
      int unary_contractions_A = extract_unary_contracted_indices(nmode_A, idx_A, nmode_B, idx_B, nmode_D, idx_D, &idx_unary_contractions_A);

      int64_t* extents_unary_contractions_A = NULL;
      extract_extents(unary_contractions_A, idx_unary_contractions_A, nmode_A, idx_A, extents_A, &extents_unary_contractions_A);

      int size_unary_contractions_A = calculate_size(extents_unary_contractions_A, unary_contractions_A);

      int64_t* idx_unary_contractions_B = NULL;
      int unary_contractions_B = extract_unary_contracted_indices(nmode_B, idx_B, nmode_A, idx_A, nmode_D, idx_D, &idx_unary_contractions_B);

      int64_t* extents_unary_contractions_B = NULL;
      extract_extents(unary_contractions_B, idx_unary_contractions_B, nmode_B, idx_B, extents_B, &extents_unary_contractions_B);

      int size_unary_contractions_B = calculate_size(extents_unary_contractions_B, unary_contractions_B);

      int64_t* strides_free_A = NULL;
      extract_free_strides(nmode_A, idx_A, strides_A, nmode_D, idx_D, &strides_free_A);

      int64_t* strides_binary_contractions_A = NULL;
      extract_contracted_strides(nmode_A, idx_A, strides_A, binary_contractions, idx_binary_contractions, &strides_binary_contractions_A);

      int64_t* strides_unary_contractions_A = NULL;
      extract_contracted_strides(nmode_A, idx_A, strides_A, unary_contractions_A, idx_unary_contractions_A, &strides_unary_contractions_A);

      int64_t* strides_free_B = NULL;
      extract_free_strides(nmode_B, idx_B, strides_B, nmode_D, idx_D, &strides_free_B);

      int64_t* strides_binary_contractions_B = NULL;
      extract_contracted_strides(nmode_B, idx_B, strides_B, binary_contractions, idx_binary_contractions, &strides_binary_contractions_B);

      int64_t* strides_unary_contractions_B = NULL;
      extract_contracted_strides(nmode_B, idx_B, strides_B, unary_contractions_B, idx_unary_contractions_B, &strides_unary_contractions_B);

      int64_t* coordinates_free = malloc(nmode_D * sizeof(int64_t));
      zero_array(coordinates_free, nmode_D);

      int64_t* coordinates_binary_contractions = malloc(binary_contractions * sizeof(int64_t));
      zero_array(coordinates_binary_contractions, binary_contractions);

      int64_t* coordinates_unary_contractions_A = malloc(unary_contractions_A * sizeof(int64_t));
      zero_array(coordinates_unary_contractions_A, unary_contractions_A);

      int64_t* coordinates_unary_contractions_B = malloc(unary_contractions_B * sizeof(int64_t));
      zero_array(coordinates_unary_contractions_B, unary_contractions_B);

      int64_t size_free = calculate_size(extents_D, nmode_D);

      void* accum = alloc_accum(prec, type_D);
      void* sum_A = alloc_val(prec, type_A);
      void* sum_B = alloc_val(prec, type_B);
      void* val_C = alloc_val(prec, type_C);
      void* prec_alpha = create_prec_scalar(alpha, type_D, prec);
      void* prec_beta = create_prec_scalar(beta, type_D, prec);

      bool is_complex_A = is_complex(type_A);
      bool is_complex_B = is_complex(type_B);
      bool is_complex_C = is_complex(type_C);
      bool is_complex_D = is_complex(type_D);

      for (int i = 0; i < size_free; i++)
      {
          int index_free_A = 0; // Index calculated from free indices of A
          int index_free_B = 0; // Index calculated from free indices of B
          int index_C = 0;
          int index_D = 0;
          for (int j = 0; j < nmode_D; j++)
          {
              index_free_A += coordinates_free[j] * strides_free_A[j];
              index_free_B += coordinates_free[j] * strides_free_B[j];
              index_C += coordinates_free[j] * strides_C[j];
              index_D += coordinates_free[j] * strides_D[j];
          }
          float val_zero = 0;
          if (!is_equal(beta, type_D, &val_zero, TAPP_F32))
          {
              get_val(val_C, C, index_C, type_C, prec);
              calculate_beta_C(prec_beta, type_D, is_complex_D, val_C, type_C, is_complex_C, op_C, prec, accum, type_D, is_complex_D);
          }
          else
          {
              zero_accum(accum, prec, type_D, is_complex_D);
          }
          for (int j = 0; j < size_binary_contractions; j++)
          {
              int index_binary_contractions_A = index_free_A;
              int index_binary_contractions_B = index_free_B;
              for (int k = 0; k < binary_contractions; k++)
              {
                  index_binary_contractions_A += coordinates_binary_contractions[k] * strides_binary_contractions_A[k];
                  index_binary_contractions_B += coordinates_binary_contractions[k] * strides_binary_contractions_B[k];
              }
              zero_sum(sum_A, prec, type_A, is_complex_A);
              for (int k = 0; k < size_unary_contractions_A; k++)
              {
                  int index_unary_contractions_A = index_binary_contractions_A;
                  for (int l = 0; l < unary_contractions_A; l++)
                  {
                      index_unary_contractions_A += coordinates_unary_contractions_A[l] * strides_unary_contractions_A[l];
                  }
                  sum_unary_contractions(sum_A, A, index_unary_contractions_A, op_A, type_A, prec);
                  increment_coordinates(coordinates_unary_contractions_A, unary_contractions_A, extents_unary_contractions_A);
              }
              zero_sum(sum_B, prec, type_B, is_complex_A);
              for (int k = 0; k < size_unary_contractions_B; k++)
              {
                  int index_unary_contractions_B = index_binary_contractions_B;
                  for (int l = 0; l < unary_contractions_B; l++)
                  {
                      index_unary_contractions_B += coordinates_unary_contractions_B[l] * strides_unary_contractions_B[l];
                  }
                  sum_unary_contractions(sum_B, B, index_unary_contractions_B, op_B, type_B, prec);
                  increment_coordinates(coordinates_unary_contractions_B, unary_contractions_B, extents_unary_contractions_B);
              }

              calculate_alpha_A_B(prec_alpha, type_D, is_complex_D, sum_A, type_A, is_complex_A, sum_B, type_B, is_complex_B, prec, accum, type_D, is_complex_D);
              increment_coordinates(coordinates_binary_contractions, binary_contractions, extents_binary_contractions);
          }
          calculate_op_D(accum, type_D, op_D, prec);
          assign_D(D, type_D, index_D, accum, prec);
          increment_coordinates(coordinates_free, nmode_D, extents_D);
      }
      free(accum);
      free(sum_A);
      free(sum_B);
      free(val_C);
      free(prec_alpha);
      free(prec_beta);
      free(idx_binary_contractions);
      free(extents_binary_contractions);
      free(strides_free_A);
      free(strides_binary_contractions_A);
      free(strides_unary_contractions_A);
      free(idx_unary_contractions_A);
      free(strides_free_B);
      free(strides_binary_contractions_B);
      free(strides_unary_contractions_B);
      free(idx_unary_contractions_B);
      free(coordinates_free);
      free(coordinates_binary_contractions);
      free(extents_unary_contractions_A);
      free(extents_unary_contractions_B);
      free(coordinates_unary_contractions_A);
      free(coordinates_unary_contractions_B);
    }

    bool comp_ = true;
    if((*exec_int_ptr) == 12 ) { // 1 = bruteforce, 2 = tblis, 12 = tblis + bruteforce check
#ifdef TAPP_REFERENCE_ENABLE_TBLIS
      comp_ = compare_tensors_(D, E_, (int64_t)size_D, type_D);
#endif
      if(!comp_){
        printf("A: \n");
        print_tensor_(nmode_A, extents_A, strides_A, A, type_D);
        printf("B: \n");
        print_tensor_(nmode_B, extents_B, strides_B, B, type_D);
        printf("C: \n");
        print_tensor_(nmode_C, extents_C, strides_C, C, type_D);
        printf("D: \n");
        print_tensor_(nmode_D, extents_D, strides_D, D, type_D);
        printf("E_: \n");
        print_tensor_(nmode_D, extents_D, strides_D, E_, type_D);
        printf("alpha: \n");
        print_tensor_(0, extents_D, strides_D, alpha, type_D);
        printf("beta: \n");
        print_tensor_(0, extents_D, strides_D, beta, type_D);
        printf("size_D: %d \n", (int)size_D);
        printf("nmode_D: %d \n", nmode_D);
      }
      free(E_);
    }

    free(idx_A);
    free(idx_B);
    free(idx_C);
    free(idx_D);
    free(extents_A);
    free(extents_B);
    free(extents_C);
    free(extents_D);
    free(strides_A);
    free(strides_B);
    free(strides_C);
    free(strides_D);

    if(!comp_) return 137;
    return 0;
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
    idx_contraction = TAPP_realloc(idx_contraction, binary_contractions * sizeof(int64_t)); //Reallocate for right amount
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
    idx_unary_contractions = TAPP_realloc(idx_unary_contractions, unary_contractions * sizeof(int64_t));
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
    float val_zero = 0;
    return tensor == NULL && !is_equal(scalar, type, &val_zero, TAPP_F32) ? error_code : 0;
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
    new_idx = TAPP_realloc(new_idx, new_nmode * sizeof(int64_t));
    new_extents = TAPP_realloc(new_extents, new_nmode * sizeof(int64_t));
    new_strides = TAPP_realloc(new_strides, new_nmode * sizeof(int64_t));
    free(*idx);
    free(*extents);
    free(*strides);
    *nmode = new_nmode;
    *idx = new_idx;
    *extents = new_extents;
    *strides = new_strides;
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

void* alloc_val(TAPP_prectype prec, TAPP_datatype type)
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

void zero_sum(void* sum, TAPP_prectype prec, TAPP_datatype type, bool is_complex)
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

void zero_accum(void* accum, TAPP_prectype prec, TAPP_datatype type, bool is_complex)
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

void calculate_beta_C(const void* beta, TAPP_datatype type_beta, bool is_complex_beta, const void* val_C, TAPP_datatype type_C, bool is_complex_C, TAPP_element_op op_C, TAPP_prectype prec, void* accum, TAPP_datatype type_accum, bool is_complex_accum)
{
    if (prec == TAPP_DEFAULT_PREC)
    {
        calculate_beta_C_default(beta, type_beta, val_C, type_C, op_C, accum, type_accum);
    }
    else
    {
        calculate_beta_C_prec(beta, is_complex_beta, val_C, is_complex_C, prec, accum, is_complex_accum);
    }
}

void calculate_beta_C_default(const void* beta, TAPP_datatype type_beta, const void* val_C, TAPP_datatype type_C, TAPP_element_op op_C, void* accum, TAPP_datatype type_accum)
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
                *(float*)accum = *(float*)beta * *(float*)val_C;
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * *(float*)val_C;
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(float*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(float*)val_C;
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
                *(float*)accum = *(float*)beta * *(double*)val_C;
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * *(double*)val_C;
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(double*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(double*)val_C;
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
                *(float*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
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
                *(float*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
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
                *(float*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(_Float16*)val_C);
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
                *(float*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_F64:
                *(float*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C32:
                *(float*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C64:
                *(float*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(float*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(float*)accum = *(__bf16*)beta * *(__bf16*)val_C;
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
                *(double*)accum = *(float*)beta * *(float*)val_C;
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * *(float*)val_C;
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(float*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(float*)val_C;
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
                *(double*)accum = *(float*)beta * *(double*)val_C;
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * *(double*)val_C;
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(double*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(double*)val_C;
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
                *(double*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
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
                *(double*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
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
                *(double*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(_Float16*)val_C);
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
                *(double*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_F64:
                *(double*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C32:
                *(double*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C64:
                *(double*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(double*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(double*)accum = *(__bf16*)beta * *(__bf16*)val_C;
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
                *(complex float*)accum = *(float*)beta * *(float*)val_C;
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(float*)val_C;
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(float*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(float*)val_C;
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
                *(complex float*)accum = *(float*)beta * *(double*)val_C;
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(double*)val_C;
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(double*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(double*)val_C;
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
                *(complex float*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
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
                *(complex float*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
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
                *(complex float*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(_Float16*)val_C);
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
                *(complex float*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_F64:
                *(complex float*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C32:
                *(complex float*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C64:
                *(complex float*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex float*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex float*)accum = *(__bf16*)beta * *(__bf16*)val_C;
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
                *(complex double*)accum = *(float*)beta * *(float*)val_C;
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(float*)val_C;
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(float*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(float*)val_C;
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
                *(complex double*)accum = *(float*)beta * *(double*)val_C;
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(double*)val_C;
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(double*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(double*)val_C;
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
                *(complex double*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
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
                *(complex double*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
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
                *(complex double*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(_Float16*)val_C);
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
                *(complex double*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_F64:
                *(complex double*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C32:
                *(complex double*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C64:
                *(complex double*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(complex double*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(complex double*)accum = *(__bf16*)beta * *(__bf16*)val_C;
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
                *(_Float16*)accum = *(float*)beta * *(float*)val_C;
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(float*)val_C;
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(float*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(float*)val_C;
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
                *(_Float16*)accum = *(float*)beta * *(double*)val_C;
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(double*)val_C;
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(double*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(double*)val_C;
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
                *(_Float16*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
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
                *(_Float16*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
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
                *(_Float16*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(_Float16*)val_C);
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
                *(_Float16*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_F64:
                *(_Float16*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C32:
                *(_Float16*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C64:
                *(_Float16*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(_Float16*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(_Float16*)accum = *(__bf16*)beta * *(__bf16*)val_C;
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
                *(__bf16*)accum = *(float*)beta * *(float*)val_C;
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(float*)val_C;
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(float*)val_C;
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(float*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(float*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(float*)val_C;
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
                *(__bf16*)accum = *(float*)beta * *(double*)val_C;
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(double*)val_C;
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(double*)val_C;
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(double*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(double*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(double*)val_C;
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
                *(__bf16*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conjf(*(complex float*)val_C) : *(complex float*)val_C);
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
                *(__bf16*)accum = *(float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * (op_C == TAPP_CONJUGATE ? conj(*(complex double*)val_C) : *(complex double*)val_C);
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
                *(__bf16*)accum = *(float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(_Float16*)val_C;
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(_Float16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(_Float16*)val_C);
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
                *(__bf16*)accum = *(float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_F64:
                *(__bf16*)accum = *(double*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C32:
                *(__bf16*)accum = *(complex float*)beta * *(__bf16*)val_C;
                break;
            case TAPP_C64:
                *(__bf16*)accum = *(complex double*)beta * *(__bf16*)val_C;
                break;
#ifdef TAPP_REFERENCE_ENABLE_F16
            case TAPP_F16:
                *(__bf16*)accum = *(_Float16*)beta * *(__bf16*)val_C;
                break;
#endif
#ifdef TAPP_REFERENCE_ENABLE_BF16
            case TAPP_BF16:
                *(__bf16*)accum = *(__bf16*)beta * *(__bf16*)val_C;
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

void calculate_beta_C_prec(const void* beta, bool is_complex_beta, const void* val_C, bool is_complex_C, TAPP_prectype prec, void* accum, bool is_complex_accum)
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
                    *(complex float*)accum = *(complex float*)beta * *(complex float*)val_C;
                }
                else
                {
                    *(complex float*)accum = *(float*)beta * *(complex float*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex float*)beta * *(float*)val_C;
                }
                else
                {
                    *(complex float*)accum = *(float*)beta * *(float*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex float*)beta * *(complex float*)val_C;
                }
                else
                {
                    *(float*)accum = *(float*)beta * *(complex float*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex float*)beta * *(float*)val_C;
                }
                else
                {
                    *(float*)accum = *(float*)beta * *(float*)val_C;
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
                    *(complex double*)accum = *(complex double*)beta * *(complex double*)val_C;
                }
                else
                {
                    *(complex double*)accum = *(double*)beta * *(complex double*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex double*)accum = *(complex double*)beta * *(double*)val_C;
                }
                else
                {
                    *(complex double*)accum = *(double*)beta * *(double*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(double*)accum = *(complex double*)beta * *(complex double*)val_C;
                }
                else
                {
                    *(double*)accum = *(double*)beta * *(complex double*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(double*)accum = *(complex double*)beta * *(double*)val_C;
                }
                else
                {
                    *(double*)accum = *(double*)beta * *(double*)val_C;
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
                    *(complex _Float16*)accum = *(complex _Float16*)beta * *(complex _Float16*)val_C;
                }
                else
                {
                    *(complex _Float16*)accum = *(_Float16*)beta * *(complex _Float16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex _Float16*)accum = *(complex _Float16*)beta * *(_Float16*)val_C;
                }
                else
                {
                    *(complex _Float16*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(_Float16*)accum = *(complex _Float16*)beta * *(complex _Float16*)val_C;
                }
                else
                {
                    *(_Float16*)accum = *(_Float16*)beta * *(complex _Float16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(_Float16*)accum = *(complex _Float16*)beta * *(_Float16*)val_C;
                }
                else
                {
                    *(_Float16*)accum = *(_Float16*)beta * *(_Float16*)val_C;
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
                    *(complex float*)accum = *(complex _Float16*)beta * *(complex _Float16*)val_C;
                }
                else
                {
                    *(complex float*)accum = *(_Float16*)beta * *(complex _Float16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex _Float16*)beta * *(_Float16*)val_C;
                }
                else
                {
                    *(complex float*)accum = *(_Float16*)beta * *(_Float16*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex _Float16*)beta * *(complex _Float16*)val_C;
                }
                else
                {
                    *(float*)accum = *(_Float16*)beta * *(complex _Float16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex _Float16*)beta * *(_Float16*)val_C;
                }
                else
                {
                    *(float*)accum = *(_Float16*)beta * *(_Float16*)val_C;
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
                    *(complex float*)accum = *(complex __bf16*)beta * *(complex __bf16*)val_C;
                }
                else
                {
                    *(complex float*)accum = *(__bf16*)beta * *(complex __bf16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(complex float*)accum = *(complex __bf16*)beta * *(__bf16*)val_C;
                }
                else
                {
                    *(complex float*)accum = *(__bf16*)beta * *(__bf16*)val_C;
                }
            }
        }
        else
        {
            if (is_complex_C)
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex __bf16*)beta * *(complex __bf16*)val_C;
                }
                else
                {
                    *(float*)accum = *(__bf16*)beta * *(complex __bf16*)val_C;
                }
            }
            else
            {
                if (is_complex_beta)
                {
                    *(float*)accum = *(complex __bf16*)beta * *(__bf16*)val_C;
                }
                else
                {
                    *(float*)accum = *(__bf16*)beta * *(__bf16*)val_C;
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

void get_val(void* val, const void* tensor, int64_t index, TAPP_datatype type, TAPP_prectype prec)
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
