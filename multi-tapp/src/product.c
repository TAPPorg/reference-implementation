#include "../include/product.h"

TAPP_error Multi_TAPP_create_tensor_product(TAPP_backend backend,
                                            TAPP_tensor_product* plan,
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
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_create_tensor_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_tensor_product\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info_A = (struct Multi_TAPP_tensor_info*)A;
    if (backend != multi_tapp_info_A->backend)
    {
        return 9; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info_B = (struct Multi_TAPP_tensor_info*)B;
    if (backend != multi_tapp_info_B->backend)
    {
        return 10; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info_C = (struct Multi_TAPP_tensor_info*)C;
    if (backend != multi_tapp_info_C->backend)
    {
        return 11; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info_D = (struct Multi_TAPP_tensor_info*)D;
    if (backend != multi_tapp_info_D->backend)
    {
        return 12; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_tensor_product* multi_tapp_plan = malloc(sizeof(struct Multi_TAPP_tensor_product));
    multi_tapp_plan->backend = backend;
    *plan = (TAPP_tensor_product)multi_tapp_plan;

    return backend_ptr->TAPP_create_tensor_product(&multi_tapp_plan->plan,
                                                         op_A, multi_tapp_info_A->info, idx_A,
                                                         op_B, multi_tapp_info_B->info, idx_B,
                                                         op_C, multi_tapp_info_C->info, idx_C,
                                                         op_D, multi_tapp_info_D->info, idx_D,
                                                         prec);
}

TAPP_error Multi_TAPP_destroy_tensor_product(TAPP_backend backend,
                                             TAPP_tensor_product plan)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_destroy_tensor_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_tensor_product\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_product* multi_tapp_plan = (struct Multi_TAPP_tensor_product*)plan;
    if (backend != multi_tapp_plan->backend)
    {
        return 13; // TODO: Return error for incompatible handle
    }

    TAPP_error error = backend_ptr->TAPP_destroy_tensor_product(multi_tapp_plan->plan);
    free(multi_tapp_plan);

    return error;
}
 
TAPP_error Multi_TAPP_execute_product(TAPP_backend backend,
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
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_execute_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_execute_product\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_product* multi_tapp_plan = (struct Multi_TAPP_tensor_product*)plan;
    if (backend != multi_tapp_plan->backend)
    {
        return 13; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_executor* multi_tapp_exec =  (struct Multi_TAPP_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    return backend_ptr->TAPP_execute_product(multi_tapp_plan->plan, multi_tapp_exec->exec, status, alpha, A, B, beta, C, D);
}

TAPP_error Multi_TAPP_execute_batched_product(TAPP_backend backend,
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
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_execute_batched_product == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_execute_batched_product\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_product* multi_tapp_plan = (struct Multi_TAPP_tensor_product*)plan;
    if (backend != multi_tapp_plan->backend)
    {
        return 13; // TODO: Return error for incompatible handle
    }

    struct Multi_TAPP_executor* multi_tapp_exec = (struct Multi_TAPP_executor*)exec;
    if (backend != multi_tapp_exec->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    return backend_ptr->TAPP_execute_batched_product(multi_tapp_plan->plan, multi_tapp_exec->exec, status, num_batches, alpha, A, B, beta, C, D);
}