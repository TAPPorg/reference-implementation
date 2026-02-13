#include "ref_impl.h"
#include "helpers.h"
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

/*
 * TODO:
 *  1. Fill in the arguments for creating the tensor (Line 47).
 *  2. Complete the function call to create the execution plan. (Line 108)
 *  3. Fill in the arguments for the execution of the product. (Line 178)
 * For a complete example usage. Look at examples/driver.c
 * Compile with: make exercise_contraction
 * The file to run is /examples/exercise_contraction/out/exercise_contraction(.exe for windows)
 */

int main(int argc, char const *argv[])
{
    // Declare handle
    TAPP_handle handle;
    TAPP_create_handle(&handle);

    /*
     * Create the tensor structures for tensor A, B, C and D.
     * Tensor A with 3 indices, with the extents 4, 3, 2, and the strides 1, 4, 12.
     * Tensor B with 3 indices, with the extents 3, 2, 4, and the strides 1, 3, 6.
     * Tensor C with 2 indices, with the extents 3, 3, and the strides 1, 3.
     * Tensor D with 2 indices, with the extents 3, 3, and the strides 1, 3.
     */

    // Tensor A
    // Assign the number of indices
    int nmode_A = 3;

    // Assign the extents
    int64_t extents_A[3] = {4, 3, 2};

    // Assign the strides
    int64_t strides_A[3] = {1, 4, 12};

    // Declare the tensor structure variable
    TAPP_tensor_info info_A;

    // Assign the structure to the variable
    /* 
     * TODO 1: Fill in the arguments for creating the tensor info.
     * Uncomment code.
     * Fill in: the tensor info object, handle, datatype(float32), structure for tensor A: number of indices, extents, strides.
     */
    //TAPP_create_tensor_info(, , , , , );

    // Tensor B
    int nmode_B = 3;
    int64_t extents_B[3] = {3, 2, 4};
    int64_t strides_B[3] = {1, 3, 6};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, handle, TAPP_F32, nmode_B, extents_B, strides_B);

    // Tensor C
    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, handle, TAPP_F32, nmode_C, extents_C, strides_C);

    // Tensor D
    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 3};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, handle, TAPP_F32, nmode_D, extents_D, strides_D);


    /*
     * Assign the options for the calculation.
     * The precision used will be the default precision.
     * The elemental operations should be the identity one (doesn't really matter since this exercise doesn't use complex numbers).
     * The operation that should be executed is:
     *  Contraction between the first index for tensor A and third index for tensor B.
     *  Contraction between the third index for tensor A and second index for tensor B.
     *  The second index for A and the first index for B are free indices, in that order. 
     */

    // Initialize the precision
    TAPP_prectype prec = TAPP_DEFAULT_PREC; 

    // Initialize the elemental operations for each of the tensors
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;

    // Create ths indicies arrays for each of the tensor
    int64_t idx_A[3] = {'a', 'b', 'c'};
    int64_t idx_B[3] = {'d', 'c', 'a'};
    int64_t idx_C[2] = {'b', 'd'};
    int64_t idx_D[2] = {'b', 'd'};

    // Declare plan
    TAPP_tensor_product plan;

    // Create plan/Assign the options to the plan
    /*
     * TODO 2: Complete the function call to create the execution plan.
     * Uncomment code.
     * Fill in: the plan, handle, computation information for tensor A and precision.
     */
    //TAPP_create_tensor_product(, , , , , op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, );

    // Declare executor
    TAPP_executor exec;

    // Create executor
    TAPP_create_executor(&exec);

    // Declare status object
    TAPP_status status;


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
    TAPP_error error;
    /* 
     * TODO 3: Fill in the arguments for the execution of the product.
     * Uncomment code.
     * Fill in: the plan, executor, status object, and the computed data: alpha, A, B, beta, C, and D.
     */
    //error = TAPP_execute_product(, , , , , , , , );


    /*
     * Print results
     */

    // Check if the execution was successful
    bool success = TAPP_check_success(error);
    
    // Print if the execution was successful
    printf(success ? "Success\n" : "Fail\n");

    // Get the length of the error message
    int message_len = TAPP_explain_error(error, 0, NULL);

    // Create a buffer to hold the message + 1 character for null terminator
    char* message_buff = malloc((message_len + 1) * sizeof(char));

    // Fetch error message
    /* 
     * TODO 4: Fill in arguments to fetch the error message from the error code to the message buffer.
     * Uncomment code.
     * Fill in: error code, message length, and buffer.
     * The length is message_len + 1 to account for null-terminator
     */
    //TAPP_explain_error(, , );

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
    TAPP_destroy_handle(handle);

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