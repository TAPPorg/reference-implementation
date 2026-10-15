#include "../include/multi_tapp/permute.h"
#include "internal.h"

TAPP_error Multi_TAPP_create_tensor_permute(Multi_TAPP_backend backend,
                                            TAPP_tensor_permute* plan,
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
    if (plan == NULL)
    {
        fprintf(stderr, "ERROR: Called function with NULL plan pointer");
        return MULTI_TAPP_ERROR_INVALID_ARGUMENT;
    }

    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_create_tensor_permute == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_tensor_permute\n");
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

    struct mt_tensor_permute* multi_tapp_plan = malloc(sizeof(struct mt_tensor_permute));
    if (multi_tapp_plan == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate space for a tensor permute plan\n");
        return MULTI_TAPP_ERROR_ALLOCATION_FAILED;
    }
    multi_tapp_plan->backend = backend;

    TAPP_error error = backend_ptr->TAPP_create_tensor_permute(&multi_tapp_plan->plan,
                                                               multi_tapp_exec->exec,
                                                               op_A, multi_tapp_info_A->info, idx_A,
                                                               op_B, multi_tapp_info_B->info, idx_B,
                                                               op_C, multi_tapp_info_C->info, idx_C,
                                                               prec);

    if (!multi_tapp_check_success(error))
    {
        free(multi_tapp_plan);
        return multi_tapp_error_from_backend(error);
    }

    *plan = (TAPP_tensor_permute)multi_tapp_plan;
    return MULTI_TAPP_SUCCESS;
}

TAPP_error Multi_TAPP_destroy_tensor_permute(Multi_TAPP_backend backend,
                                             TAPP_tensor_permute plan)
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
    if (backend_ptr->TAPP_destroy_tensor_permute == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_tensor_permute\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_tensor_permute* multi_tapp_plan = (struct mt_tensor_permute*)plan;
    if (backend != multi_tapp_plan->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_PERMUTE_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_destroy_tensor_permute(multi_tapp_plan->plan);
    free(multi_tapp_plan);

    return multi_tapp_error_from_backend(error);
}
 
TAPP_error Multi_TAPP_execute_permute(Multi_TAPP_backend backend,
                                      TAPP_tensor_permute plan,
                                      TAPP_executor exec,
                                      TAPP_status* status,
                                      const void* alpha,
                                      const void* A,
                                      const void* beta,
                                      const void* B,
                                            void* C)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (plan == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor permute plan\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_PERMUTE;
    }
    if (exec == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL executor\n");
        return MULTI_TAPP_ERROR_INVALID_EXECUTOR;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_execute_permute == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_execute_permute\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_tensor_permute* multi_tapp_plan = (struct mt_tensor_permute*)plan;
    if (backend != multi_tapp_plan->backend)
    {
        return MULTI_TAPP_ERROR_TENSOR_PERMUTE_WRONG_BACKEND;
    }

    struct mt_executor* multi_tapp_exec =  (struct mt_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND;
    }

    struct mt_status* multi_tapp_status;
    TAPP_error error = multi_tapp_status_wrap(backend, status, &multi_tapp_status);
    if (error != MULTI_TAPP_SUCCESS) return error;

    error = backend_ptr->TAPP_execute_permute(multi_tapp_plan->plan, multi_tapp_exec->exec, multi_tapp_status_of(multi_tapp_status),
                                              alpha, A, beta, B, C);

    return multi_tapp_status_confirm(multi_tapp_status, status, error);
}