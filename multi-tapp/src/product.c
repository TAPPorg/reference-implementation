#include "../include/product.h"

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
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_create_tensor_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_tensor_product\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info_A = (struct Multi_TAPP_tensor_info*)A;
    if (multi_tapp_handle->impl_id != multi_tapp_info_A->impl_id)
    {
        return 9; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info_B = (struct Multi_TAPP_tensor_info*)B;
    if (multi_tapp_handle->impl_id != multi_tapp_info_B->impl_id)
    {
        return 10; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info_C = (struct Multi_TAPP_tensor_info*)C;
    if (multi_tapp_handle->impl_id != multi_tapp_info_C->impl_id)
    {
        return 11; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info_D = (struct Multi_TAPP_tensor_info*)D;
    if (multi_tapp_handle->impl_id != multi_tapp_info_D->impl_id)
    {
        return 12; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_tensor_product* multi_tapp_plan = malloc(sizeof(struct Multi_TAPP_tensor_product));
    multi_tapp_plan->impl_id = multi_tapp_handle->impl_id;
    *plan = (TAPP_tensor_product)multi_tapp_plan;

    return multi_tapp_handle->TAPP_create_tensor_product(multi_tapp_plan->plan,
                                                         *multi_tapp_handle->tapp_handle,
                                                         op_A, *multi_tapp_info_A->info, idx_A,
                                                         op_B, *multi_tapp_info_B->info, idx_B,
                                                         op_C, *multi_tapp_info_C->info, idx_C,
                                                         op_D, *multi_tapp_info_D->info, idx_D,
                                                         prec);
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan,
                                       TAPP_handle handle)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_destroy_tensor_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_tensor_product\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_product* multi_tapp_plan = (struct Multi_TAPP_tensor_product*)plan;
    if (multi_tapp_handle->impl_id != multi_tapp_plan->impl_id)
    {
        return 13; // TODO: Return error for incompatible handle
    }

    TAPP_error error = multi_tapp_handle->TAPP_destroy_tensor_product(*multi_tapp_plan->plan, *multi_tapp_handle->tapp_handle);
    free(multi_tapp_plan);

    return error;
}
 
TAPP_error TAPP_execute_product(TAPP_tensor_product plan,
                                TAPP_handle handle,
                                TAPP_executor exec,
                                TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* B,
                                const void* beta,
                                const void* C,
                                      void* D)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_execute_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_execute_product\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_product* multi_tapp_plan = (struct Multi_TAPP_tensor_product*)plan;
    if (multi_tapp_handle->impl_id != multi_tapp_plan->impl_id)
    {
        return 13; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_executor* multi_tapp_exec =  (struct Multi_TAPP_executor*)exec;
    if (multi_tapp_handle->impl_id != multi_tapp_exec->impl_id)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_status* multi_tapp_status = malloc(sizeof(struct Multi_TAPP_status));
    multi_tapp_status->impl_id = multi_tapp_handle->impl_id;
    *status = (TAPP_status)multi_tapp_status;

    return multi_tapp_handle->TAPP_execute_product(*multi_tapp_plan->plan, *multi_tapp_handle->tapp_handle, *multi_tapp_exec->exec, multi_tapp_status->status, alpha, A, B, beta, C, D);
}

TAPP_error TAPP_execute_batched_product(TAPP_tensor_product plan,
                                        TAPP_handle handle,
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
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_execute_batched_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_execute_batched_product\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_product* multi_tapp_plan = (struct Multi_TAPP_tensor_product*)plan;
    if (multi_tapp_handle->impl_id != multi_tapp_plan->impl_id)
    {
        return 13; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_executor* multi_tapp_exec = (struct Multi_TAPP_executor*)exec;
    if (multi_tapp_handle->impl_id != multi_tapp_exec->impl_id)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_status* multi_tapp_status = malloc(sizeof(struct Multi_TAPP_status));
    multi_tapp_status->impl_id = multi_tapp_handle->impl_id;
    *status = (TAPP_status)multi_tapp_status;

    return multi_tapp_handle->TAPP_execute_batched_product(*multi_tapp_plan->plan, *multi_tapp_handle->tapp_handle, *multi_tapp_exec->exec, multi_tapp_status->status, num_batches, alpha, A, B, beta, C, D);
}