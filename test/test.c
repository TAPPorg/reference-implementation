/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#include "hi_tapp/product.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{
    int nmode_A = 3;
    int64_t extents_A[3] = {4, 3, 3};
    int64_t strides_A[3] = {1, 4, 12};
    HI_TAPP_tensor_info info_A;
    HI_TAPP_create_tensor_info(&info_A, HI_TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    HI_TAPP_tensor_info info_B;
    HI_TAPP_create_tensor_info(&info_B, HI_TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {4, 2};
    int64_t strides_C[2] = {1, 4};
    HI_TAPP_tensor_info info_C;
    HI_TAPP_create_tensor_info(&info_C, HI_TAPP_F32, nmode_C, extents_C, strides_C);
    
    int nmode_D = 2;
    int64_t extents_D[2] = {4, 2};
    int64_t strides_D[2] = {1, 4};
    HI_TAPP_tensor_info info_D;
    HI_TAPP_create_tensor_info(&info_D, HI_TAPP_F32, nmode_D, extents_D, strides_D);

    HI_TAPP_handle handle;
    HI_TAPP_tensor_product plan;
    HI_TAPP_element_op op_A = HI_TAPP_IDENTITY;
    HI_TAPP_element_op op_B = HI_TAPP_IDENTITY;
    HI_TAPP_element_op op_C = HI_TAPP_IDENTITY;
    HI_TAPP_element_op op_D = HI_TAPP_IDENTITY;
    int64_t idx_A[3] = {'a', 'b', 'c'};
    int64_t idx_B[4] = {'c', 'd', 'e', 'b'};
    int64_t idx_C[2] = {'a', 'd'};
    int64_t idx_D[3] = {'a', 'd'};
    HI_TAPP_prectype prec = HI_TAPP_DEFAULT_PREC;
    HI_TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    HI_TAPP_executor exec;
    create_executor(&exec);
    HI_TAPP_status status;

    float alpha = 1;

    float A[36] = {
        1,  2,  1.01, -1,
        1,  2,  1.01, -1,
        1,  2,  1.01, -1,

        1,  2,  1.01, -1,
        1,  2,  1.01, -1,
        1,  2,  1.01, -1,

        1,  2,  1.01, -1,
        1,  2,  1.01, -1,
        1,  2,  1.01, -1
    };

    float B[36] = {
        1,  1,  1,
        2,  2,  2,

        3,  3,  3,
        6,  6,  6,


        1,  1,  1,
        2,  2,  2,

        3,  3,  3,
        6,  6,  6,


        1,  1,  1,
        2,  2,  2,

        3,  3,  3,
        6,  6,  6
    };

    float beta = 0;

    float C[16] = {
        2,  4,  6,  8,
        2,  4,  6,  8,

        2,  4,  6,  8,
        2,  4,  6,  8
    };

    float D[16] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
        
         1,  2,  3,  4,
         5,  6,  7,  8
    };

    HI_TAPP_error error = HI_TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);
    printf(HI_TAPP_check_success(error) ? "Success\n" : "Fail\n");
    int message_len = HI_TAPP_explain_error(error, 0, NULL);
    char* message_buff = malloc((message_len + 1) * sizeof(char));
    HI_TAPP_explain_error(error, message_len + 1, message_buff);
    printf(message_buff);
    free(message_buff);

    HI_TAPP_destory_tensor_product(plan);
    HI_TAPP_destory_tensor_info(info_A);
    HI_TAPP_destory_tensor_info(info_B);
    HI_TAPP_destory_tensor_info(info_C);
    HI_TAPP_destory_tensor_info(info_D);
    return 0;
}
