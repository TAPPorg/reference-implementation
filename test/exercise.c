#include <tapp.h>
#include "helpers.h"
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

int main(int argc, char const *argv[])
{
    /*
     * Create the tensor structures for tensor A, B, C and D.
     * Tensor A 3 dimensional tensor with the extents 4, 3, 2, and the strides 1, 4, 12.
     * Tensor B 3 dimensional tensor with the extents 3, 2, 4, and the strides 1, 3, 6.
     * Tensor C 2 dimensional tensor with the extents 3, 3, and the strides 1, 3.
     * Tensor D 2 dimensional tensor with the extents 3, 3, and the strides 1, 3.
     */

    // Tensor A
    // Assign the number of indices
    /* Remove */ int nmode_A = 3;

    // Assign the extents
    /* Remove */ int64_t extents_A[3] = {4, 3, 2};

    // Assign the strides
    /* Remove */ int64_t strides_A[3] = {1, 4, 12};

    // Declare the tensor structure variable
    /* Remove */ TAPP_tensor_info info_A;

    // Assign the structure to the variable
    /* Remove */ TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    // Tensor B
    /* Remove */ int nmode_B = 3;
    /* Remove */ int64_t extents_B[3] = {3, 2, 4};
    /* Remove */ int64_t strides_B[3] = {1, 3, 6};
    /* Remove */ TAPP_tensor_info info_B;
    /* Remove */ TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    // Tensor C
    /* Remove */ int nmode_C = 2;
    /* Remove */ int64_t extents_C[2] = {3, 3};
    /* Remove */ int64_t strides_C[2] = {1, 3};
    /* Remove */ TAPP_tensor_info info_C;
    /* Remove */ TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    // Tensor D
    /* Remove */ int nmode_D = 2;
    /* Remove */ int64_t extents_D[2] = {3, 3};
    /* Remove */ int64_t strides_D[2] = {1, 3};
    /* Remove */ TAPP_tensor_info info_D;
    /* Remove */ TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    /*
     * Assign the options for the calculation.
     * The precision used will be the default precision.
     * The elemental operations should be the identity one (doesn't really matter since this exercise doesn't use complex numbers).
     * The operation that should be executed is:
     *  Contraction between the first index for tensor A and third index for tensor B.
     *  Contraction between the third index for tensor A and second index for tensor B.
     *  The second index for A and the first index for B are free indices, in that order. 
     */

    // Declare handle (no assignment)
    /* Remove */ TAPP_handle handle;

    // Initialize the precision
    /* Remove */ TAPP_prectype prec = TAPP_DEFAULT_PREC; 

    // Initialize the elemental operations for each of the tensors
    /* Remove */ TAPP_element_op op_A = TAPP_IDENTITY;
    /* Remove */ TAPP_element_op op_B = TAPP_IDENTITY;
    /* Remove */ TAPP_element_op op_C = TAPP_IDENTITY;
    /* Remove */ TAPP_element_op op_D = TAPP_IDENTITY;

    // Create ths indicies arrays for each of the tensor
    /* Remove */ int64_t idx_A[3] = {'a', 'b', 'c'};
    /* Remove */ int64_t idx_B[3] = {'d', 'c', 'a'};
    /* Remove */ int64_t idx_C[2] = {'b', 'd'};
    /* Remove */ int64_t idx_D[2] = {'b', 'd'};

    // Declare plan
    /* Remove */ TAPP_tensor_product plan;

    // Create plan/Assign the options to the plan
    /* Remove */ TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    // Declare executor
    /* Remove */ TAPP_executor exec;

    // Create executor
    TAPP_create_executor(&exec);

    // Declare status object
    /* Remove */ TAPP_status status;


    /*
     * Assign data for the execution
     */
    
    // Initialize alpha
    float alpha = 3;

    // Initialize data for tensor A
    float A[24] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    // Initialize data for tensor B
    float B[24] = {
        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6};

    // Initialize beta
    float beta = 2;

    // Initialize data for tensor C
    float C[9] = {
        4, 4, 8,
        4, 8, 8,
        8, 8, 8};

    // Initialize data for tensor D
    float D[9] = {
        2, 3, 4,
        5, 6, 7,
        9, 1, 2};
    

    /*
     * Run the execution
     */

    // Call the execution function
    /* Remove */TAPP_error error = TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);


    /*
     * Print results
     */

    // Check if the execution was successful
    bool success = /* Remove */ TAPP_check_success(error);
    
    // Print if the execution was successful
    printf(success ? "Success\n" : "Fail\n");

    // Get the length of the error message
    /* Remove */ int message_len = TAPP_explain_error(error, 0, NULL);

    // Create a buffer to hold the message + 1 character for null terminator
    /* Remove */ char* message_buff = malloc((message_len + 1) * sizeof(char));

    // Fetch error message
    /* Remove */ TAPP_explain_error(error, message_len + 1, message_buff);

    // Print error message
    printf("%s", message_buff);
    printf("\n");

    // Print the output
    print_tensor_s(nmode_D, extents_D, strides_D, D);
    

    /*
     * Free data
     */

    // Free buffer
    free(message_buff);

    // Destroy structures
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);

    /*
     * Expected output:
    Success
    Success.
        53.090 53.090 61.090 
        53.090 61.090 61.090 
        61.090 61.090 61.090 
     */

    return 0;
}
