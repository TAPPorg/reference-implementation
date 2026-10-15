#include "../include/multi_tapp/product.h"
#include "internal.h"

TAPP_error Multi_TAPP_create_tensor_product(Multi_TAPP_backend backend,
                                            TAPP_tensor_product* plan,
                                            TAPP_executor exec,
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
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (exec == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL executor\n");
        return MULTI_TAPP_ERROR_INVALID_EXECUTOR;
    }
    if (A == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info A\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_A;
    }
    if (B == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info B\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_B;
    }
    if (C == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info C\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_C;
    }
    if (D == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info D\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_INFO_D;
    }
    if (plan == NULL)
    {
        fprintf(stderr, "ERROR: Called function with NULL plan pointer");
        return MULTI_TAPP_ERROR_INVALID_ARGUMENT;
    }

    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_create_tensor_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_tensor_product\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_executor* multi_tapp_exec = (struct mt_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND;
    }

    struct mt_tensor_info* multi_tapp_info_A = (struct mt_tensor_info*)A;
    if (backend != multi_tapp_info_A->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_INFO_A_WRONG_BACKEND;
    }

    struct mt_tensor_info* multi_tapp_info_B = (struct mt_tensor_info*)B;
    if (backend != multi_tapp_info_B->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_INFO_B_WRONG_BACKEND;
    }

    struct mt_tensor_info* multi_tapp_info_C = (struct mt_tensor_info*)C;
    if (backend != multi_tapp_info_C->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_INFO_C_WRONG_BACKEND;
    }

    struct mt_tensor_info* multi_tapp_info_D = (struct mt_tensor_info*)D;
    if (backend != multi_tapp_info_D->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_INFO_D_WRONG_BACKEND;
    }

    struct mt_tensor_product* multi_tapp_plan = malloc(sizeof(struct mt_tensor_product));
    if (multi_tapp_plan == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate space for a tensor product plan\n");
        return MULTI_TAPP_ERROR_ALLOCATION_FAILED;
    }
    multi_tapp_plan->backend = backend;

    TAPP_error error = backend_ptr->TAPP_create_tensor_product(&multi_tapp_plan->plan,
                                                               multi_tapp_exec->exec,
                                                               op_A, multi_tapp_info_A->info, idx_A,
                                                               op_B, multi_tapp_info_B->info, idx_B,
                                                               op_C, multi_tapp_info_C->info, idx_C,
                                                               op_D, multi_tapp_info_D->info, idx_D,
                                                               prec);

    if (!multi_tapp_check_success(error))
    {
        free(multi_tapp_plan);
        return multi_tapp_error_from_backend(error);
    }

    *plan = (TAPP_tensor_product)multi_tapp_plan;
    return MULTI_TAPP_SUCCESS;
}

TAPP_error Multi_TAPP_destroy_tensor_product(Multi_TAPP_backend backend,
                                             TAPP_tensor_product plan)
{
    if (plan == 0)
    {
        fprintf(stderr, "WARNING: Tried to destroy plan 0\n");
        return MULTI_TAPP_SUCCESS;
    }
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_destroy_tensor_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_tensor_product\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_tensor_product* multi_tapp_plan = (struct mt_tensor_product*)plan;
    if (backend != multi_tapp_plan->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_PRODUCT_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_destroy_tensor_product(multi_tapp_plan->plan);
    free(multi_tapp_plan);

    return multi_tapp_error_from_backend(error);
}
 
TAPP_error Multi_TAPP_execute_product(Multi_TAPP_backend backend,
                                      TAPP_tensor_product plan,
                                      TAPP_executor exec,
                                      TAPP_status* status,
                                      const void* alpha,
                                      const void* A,
                                      const void* B,
                                      const void* beta,
                                      const void* C,
                                            void* D)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (plan == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor product plan\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_PRODUCT;
    }
    if (exec == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL executor\n");
        return MULTI_TAPP_ERROR_INVALID_EXECUTOR;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_execute_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_execute_product\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_tensor_product* multi_tapp_plan = (struct mt_tensor_product*)plan;
    if (backend != multi_tapp_plan->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_PRODUCT_WRONG_BACKEND;
    }

    struct mt_executor* multi_tapp_exec =  (struct mt_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND;
    }

    struct mt_status* multi_tapp_status;
    TAPP_error error = multi_tapp_status_wrap(backend, status, &multi_tapp_status);
    if (error != MULTI_TAPP_SUCCESS) return error;

    error = backend_ptr->TAPP_execute_product(multi_tapp_plan->plan, multi_tapp_exec->exec, multi_tapp_status_of(multi_tapp_status),
                                              alpha, A, B, beta, C, D);

    return multi_tapp_status_confirm(multi_tapp_status, status, error);
}

TAPP_error Multi_TAPP_execute_batched_product(Multi_TAPP_backend backend,
                                              TAPP_tensor_product plan,
                                              TAPP_executor exec,
                                              TAPP_status* status,
                                              int num_batches,
                                              const void* alpha,
                                              const void** A,
                                              const void** B,
                                              const void* beta,
                                              const void** C,
                                                    void** D)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (plan == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor product plan\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_PRODUCT;
    }
    if (exec == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL executor\n");
        return MULTI_TAPP_ERROR_INVALID_EXECUTOR;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_execute_batched_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_execute_batched_product\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_tensor_product* multi_tapp_plan = (struct mt_tensor_product*)plan;
    if (backend != multi_tapp_plan->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_PRODUCT_WRONG_BACKEND;
    }

    struct mt_executor* multi_tapp_exec = (struct mt_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND;
    }

    struct mt_status* multi_tapp_status;
    TAPP_error error = multi_tapp_status_wrap(backend, status, &multi_tapp_status);
    if (error != MULTI_TAPP_SUCCESS) return error;

    error = backend_ptr->TAPP_execute_batched_product(multi_tapp_plan->plan, multi_tapp_exec->exec, multi_tapp_status_of(multi_tapp_status),
                                                      num_batches, alpha, A, B, beta, C, D);

    return multi_tapp_status_confirm(multi_tapp_status, status, error);
}