#include "benchmark.h"

void load_implementation()
{
    imp.handle = dlopen(imp_path, RTLD_LAZY);
    if (!imp.handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return;
    }
    dlerror();
    *(void**)(&imp.TAPP_attr_set) = dlsym(imp.handle, "TAPP_attr_set");
    *(void**)(&imp.TAPP_attr_get) = dlsym(imp.handle, "TAPP_attr_get");
    *(void**)(&imp.TAPP_attr_clear) = dlsym(imp.handle, "TAPP_attr_clear");
    *(void**)(&imp.TAPP_check_success) = dlsym(imp.handle, "TAPP_check_success");
    *(void**)(&imp.TAPP_explain_error) = dlsym(imp.handle, "TAPP_explain_error");
    *(void**)(&imp.create_executor) = dlsym(imp.handle, "create_executor");
    *(void**)(&imp.TAPP_destroy_executor) = dlsym(imp.handle, "TAPP_destroy_executor");
    *(void**)(&imp.create_handle) = dlsym(imp.handle, "create_handle");
    *(void**)(&imp.TAPP_destroy_handle) = dlsym(imp.handle, "TAPP_destroy_handle");
    *(void**)(&imp.TAPP_create_tensor_product) = dlsym(imp.handle, "TAPP_create_tensor_product");
    *(void**)(&imp.TAPP_destroy_tensor_product) = dlsym(imp.handle, "TAPP_destroy_tensor_product");
    *(void**)(&imp.TAPP_execute_product) = dlsym(imp.handle, "TAPP_execute_product");
    *(void**)(&imp.TAPP_execute_batched_product) = dlsym(imp.handle, "TAPP_execute_batched_product");
    *(void**)(&imp.TAPP_destroy_status) = dlsym(imp.handle, "TAPP_destroy_status");
    *(void**)(&imp.TAPP_create_tensor_info) = dlsym(imp.handle, "TAPP_create_tensor_info");
    *(void**)(&imp.TAPP_destroy_tensor_info) = dlsym(imp.handle, "TAPP_destroy_tensor_info");
    *(void**)(&imp.TAPP_get_nmodes) = dlsym(imp.handle, "TAPP_get_nmodes");
    *(void**)(&imp.TAPP_set_nmodes) = dlsym(imp.handle, "TAPP_set_nmodes");
    *(void**)(&imp.TAPP_get_extents) = dlsym(imp.handle, "TAPP_get_extents");
    *(void**)(&imp.TAPP_set_extents) = dlsym(imp.handle, "TAPP_set_extents");
    *(void**)(&imp.TAPP_get_strides) = dlsym(imp.handle, "TAPP_get_strides");
    *(void**)(&imp.TAPP_set_strides) = dlsym(imp.handle, "TAPP_set_strides");
    const char* error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "dlsym failed: %s\n", error);
        dlclose(imp.handle);
        return;
    }
}

void unload_implementation()
{
    if (imp.handle) {
        dlclose(imp.handle);
        imp.handle = NULL;
    }
}

void load_strides()
{
    strides_list[48][TENSOR_A] = malloc(sizeof(int64_t) * 3);
    strides_list[48][TENSOR_A][0] = 2; strides_list[48][TENSOR_A][1] = 384*2*2; strides_list[48][TENSOR_A][2] = 384*384*2*2*2;
    strides_list[48][TENSOR_B] = malloc(sizeof(int64_t) * 3);
    strides_list[48][TENSOR_B][0] = 2; strides_list[48][TENSOR_B][1] = 384*2*2; strides_list[48][TENSOR_B][2] = 384*384*2*2*2;
    strides_list[48][TENSOR_D] = malloc(sizeof(int64_t) * 2);
    strides_list[48][TENSOR_D][0] = 2; strides_list[48][TENSOR_D][1] = 384*2*2;
}

void unload_strides()
{
    free(strides_list[48][TENSOR_A]);
    free(strides_list[48][TENSOR_B]);
    free(strides_list[48][TENSOR_D]);
}

int* count_nmodes(int test_id)
{
    int* nmodes = malloc(sizeof(int) * NUMBER_OF_TENSOR_INFOS);
    for (int i = 0; i < NUMBER_OF_TENSOR_INFOS; i++)
    {
        nmodes[i] = 0;
    }
    
    int working_tensor = 0;
    for (int i = 0; indices_list[test_id][i] != '\0'; i++)
    {
        if (indices_list[test_id][i] == '-')
        {
            working_tensor++;
            continue;
        }
        nmodes[working_tensor]++;
    }
    return nmodes;
}

TAPP_tensor_info** create_tensor_infos(int test_id)
{
    int* nmodes = count_nmodes(test_id);

    int64_t* extents[NUMBER_OF_TENSOR_INFOS] = {
        malloc(sizeof(int64_t) * nmodes[TENSOR_A]),
        malloc(sizeof(int64_t) * nmodes[TENSOR_B]),
        malloc(sizeof(int64_t) * nmodes[TENSOR_D])
    };
    int64_t* strides[NUMBER_OF_TENSOR_INFOS] = {
        malloc(sizeof(int64_t) * nmodes[TENSOR_A]),
        malloc(sizeof(int64_t) * nmodes[TENSOR_B]),
        malloc(sizeof(int64_t) * nmodes[TENSOR_D])
    };

    int working_tensor = 0;
    int working_index = 0;
    int stride = 1;
    for (int i = 0; indices_list[test_id][i] != '\0'; i++)
    {
        if (indices_list[test_id][i] == '-'){
            working_tensor++;
            working_index = 0;
            stride = 1;
            continue;
        }
        extents[working_tensor][working_index] = extents_list[test_id][(int)(indices_list[test_id][i]) - (int)'a'];
        strides[working_tensor][working_index] = stride;
        stride *= extents[working_tensor][working_index];
        working_index++;
    }

    if (strides_list[test_id][TENSOR_A] != NULL) memcpy(strides[TENSOR_A], strides_list[test_id][TENSOR_A], sizeof(int64_t) * nmodes[TENSOR_A]);
    if (strides_list[test_id][TENSOR_B] != NULL) memcpy(strides[TENSOR_B], strides_list[test_id][TENSOR_B], sizeof(int64_t) * nmodes[TENSOR_B]);
    if (strides_list[test_id][TENSOR_D] != NULL) memcpy(strides[TENSOR_D], strides_list[test_id][TENSOR_D], sizeof(int64_t) * nmodes[TENSOR_D]);

    TAPP_tensor_info** tensors = malloc(sizeof(TAPP_tensor_info*) * NUMBER_OF_TENSOR_INFOS);

    for (int i = 0; i < NUMBER_OF_TENSOR_INFOS; i++)
    {
        tensors[i] = malloc(sizeof(TAPP_tensor_info));
        imp.TAPP_create_tensor_info(tensors[i], datatype_list[test_id][i], nmodes[i], extents[i], strides[i]);
        
        free(extents[i]);
        free(strides[i]);
    }

    free(nmodes);

    return tensors;
}

int64_t** create_indices(int test_id, TAPP_tensor_info** tensors)
{
    int64_t** indices = malloc(sizeof(int64_t*) * NUMBER_OF_TENSOR_INFOS);
    for (int i = 0; i < NUMBER_OF_TENSOR_INFOS; i++)
    {
        indices[i] = malloc(sizeof(int64_t) * imp.TAPP_get_nmodes(*tensors[i]));
    }
    
    int working_tensor = 0;
    int working_index = 0;
    for (int i = 0; indices_list[test_id][i] != '\0'; i++)
    {
        if (indices_list[test_id][i] == '-'){
            working_tensor++;
            working_index = 0;
            continue;
        }
        indices[working_tensor][working_index] = (int64_t)indices_list[test_id][i];
        working_index++;
    }
    return indices;
}

float randf(float min, float max)
{
    return ((float)rand()/(float)(RAND_MAX)) * (max - min) + min;
}

double randd(double min, double max)
{
    return ((double)rand()/(double)(RAND_MAX)) * (max - min) + min;
}

void* create_random_values(int64_t number_of_values, TAPP_datatype datatype)
{
    if (datatype == TAPP_FLOAT)
    {
        float* values = malloc(sizeof(float) * number_of_values);
        for (int64_t i = 0; i < number_of_values; i++)
        {
            values[i] = randf(-10, 10);
        }
        return values;
    }
    else if (datatype == TAPP_DOUBLE)
    {
        double* values = malloc(sizeof(double) * number_of_values);
        for (int64_t i = 0; i < number_of_values; i++)
        {
            values[i] = randd(-10, 10);
        }
        return values;
    }
    else if (datatype == TAPP_SCOMPLEX)
    {
        complex float* values = malloc(sizeof(complex float) * number_of_values);
        for (int64_t i = 0; i < number_of_values; i++)
        {
            values[i] = randf(-10, 10) + randf(-10, 10) * I;
        }
        return values;
    }
    else if (datatype == TAPP_DCOMPLEX)
    {
        complex double* values = malloc(sizeof(complex double) * number_of_values);
        for (int64_t i = 0; i < number_of_values; i++)
        {
            values[i] = randd(-10, 10) + randd(-10, 10) * I;
        }
        return values;
    }
    return NULL;
}

void* create_tensor_values(TAPP_tensor_info tensor, TAPP_datatype datatype)
{
    int64_t size = 1;
    int64_t* extents = malloc(sizeof(int64_t) * imp.TAPP_get_nmodes(tensor));
    int64_t* strides = malloc(sizeof(int64_t) * imp.TAPP_get_nmodes(tensor));
    imp.TAPP_get_extents(tensor, extents);
    imp.TAPP_get_strides(tensor, strides);

    for (int i = 0; i < imp.TAPP_get_nmodes(tensor); i++)
    {
        size += abs(strides[i] * (extents[i] - 1));
    }

    free(extents);
    free(strides);

    return create_random_values(size, datatype);
}

intptr_t calculate_offset(TAPP_tensor_info tensor, TAPP_datatype datatype)
{
    intptr_t offset = 0;
    int64_t* extents = malloc(sizeof(int64_t) * imp.TAPP_get_nmodes(tensor));
    int64_t* strides = malloc(sizeof(int64_t) * imp.TAPP_get_nmodes(tensor));
    imp.TAPP_get_extents(tensor, extents);
    imp.TAPP_get_strides(tensor, strides);

    for (int i = 0; i < imp.TAPP_get_nmodes(tensor); i++)
    {
        if (strides[i] < 0)
        {
            offset += abs(strides[i] * (extents[i] - 1));
        }
    }

    if (datatype == TAPP_FLOAT)
    {
        offset *= sizeof(float);
    }
    else if (datatype == TAPP_DOUBLE)
    {
        offset *= sizeof(double);
    }
    else if (datatype == TAPP_SCOMPLEX)
    {
        offset *= sizeof(complex float);
    }
    else if (datatype == TAPP_DCOMPLEX)
    {
        offset *= sizeof(complex double);
    }
    
    free(extents);
    free(strides);

    return offset;
}

uint64_t calculate_FLO(int test_id)
{
    uint64_t unary_contraction_A_size = 1;
    uint64_t unary_contraction_B_size = 1;
    uint64_t free_A_size = 1;
    uint64_t free_B_size = 1;
    uint64_t binary_contraction_size = 1;
    uint64_t result_size = 1;
    for (int i = 0, current_tensor = TENSOR_A; current_tensor < TENSOR_D; i++)
    {
        if (indices_list[test_id][i] == '-'){
            current_tensor++;
            continue;
        }
        int test_tensor = TENSOR_A;
        bool in_other = false;
        bool in_result = false;
        for (int j = 0; indices_list[test_id][j] != '\0'; j++)
        {
            if (indices_list[test_id][j] == '-')
            {
                test_tensor++;
                continue;
            }
            if (indices_list[test_id][i] == indices_list[test_id][j] && j < i)
            {
                goto skip;
            }
            if (current_tensor == test_tensor || current_tensor == TENSOR_D)
            {
                continue;
            }
            if (indices_list[test_id][i] == indices_list[test_id][j] && current_tensor != test_tensor && test_tensor != TENSOR_D)
            {
                in_other = true;
            }
            if (indices_list[test_id][i] == indices_list[test_id][j] && test_tensor == TENSOR_D)
            {
                in_result = true;
            }
        }

        if (!in_result && !in_other)
        {
            if (current_tensor == TENSOR_A)
            {
                unary_contraction_A_size *= extents_list[test_id][(int)(indices_list[test_id][i]) - (int)'a'];
            }
            else if (current_tensor == TENSOR_B)
            {
                unary_contraction_B_size *= extents_list[test_id][(int)(indices_list[test_id][i]) - (int)'a'];
            }
        }
        else if (in_other && !in_result)
        {
            binary_contraction_size *= extents_list[test_id][(int)(indices_list[test_id][i]) - (int)'a'];
        }
        else if (in_result)
        {
            result_size *= extents_list[test_id][(int)(indices_list[test_id][i]) - (int)'a'];
            if (current_tensor == TENSOR_A)
            {
                free_A_size *= extents_list[test_id][(int)(indices_list[test_id][i]) - (int)'a'];
            }
            else if (current_tensor == TENSOR_B)
            {
                free_B_size *= extents_list[test_id][(int)(indices_list[test_id][i]) - (int)'a'];
            }
        }
        
        skip:
    }
    
    uint64_t FLO = 2 * result_size; // beta * C

    FLO += (unary_contraction_A_size - 1) * free_A_size; // unary contractions for A (more optimized than reference implementation)

    FLO += (unary_contraction_B_size - 1) * free_B_size; // unary contractions for B (more optimized than reference implementation)
    
    FLO += result_size * binary_contraction_size * 2; // A * B

    FLO += result_size; // alpha

    return FLO;
}

int main(int argc, char const *argv[])
{
    load_implementation(&imp);
    load_strides();
    for (int test_id = 0; test_id < NUMBER_OF_TESTS; test_id++)
    {
        printf("Test %d, %s: ", test_id + 1, indices_list[test_id]);
        fflush(stdout);
        TAPP_tensor_info** tensors = create_tensor_infos(test_id);
        int64_t** indices = create_indices(test_id, tensors);
        TAPP_handle handle;
        imp.create_handle(&handle);
        TAPP_tensor_product plan;
        imp.TAPP_create_tensor_product(&plan, handle, op_list[test_id][OP_A], *tensors[TENSOR_A], indices[TENSOR_A], op_list[test_id][OP_B], *tensors[TENSOR_B], indices[TENSOR_B], op_list[test_id][OP_C], *tensors[TENSOR_C], indices[TENSOR_C], op_list[test_id][OP_D], *tensors[TENSOR_D], indices[TENSOR_D], precision_list[test_id]);
        TAPP_executor exec;
        imp.create_executor(&exec);
        TAPP_status status;
        void* alpha = create_random_values(1, datatype_list[test_id][DATATYPE_ALPHA]);
        void* beta = create_random_values(1, datatype_list[test_id][DATATYPE_BETA]);
        void* A = create_tensor_values(*tensors[TENSOR_A], datatype_list[test_id][DATATYPE_A]);
        void* B = create_tensor_values(*tensors[TENSOR_B], datatype_list[test_id][DATATYPE_B]);
        void* C = create_tensor_values(*tensors[TENSOR_C], datatype_list[test_id][DATATYPE_C]);
        void* D = create_tensor_values(*tensors[TENSOR_D], datatype_list[test_id][DATATYPE_D]);
        intptr_t offset_A = calculate_offset(*tensors[TENSOR_A], datatype_list[test_id][DATATYPE_A]);
        intptr_t offset_B = calculate_offset(*tensors[TENSOR_B], datatype_list[test_id][DATATYPE_B]);
        intptr_t offset_C = calculate_offset(*tensors[TENSOR_C], datatype_list[test_id][DATATYPE_C]);
        intptr_t offset_D = offset_C;
        uint64_t FLO = calculate_FLO(test_id);
        clock_t start = clock();
        TAPP_error error = imp.TAPP_execute_product(plan, exec, &status, (void *)alpha, (void *)((intptr_t)A + offset_A), (void *)((intptr_t)B + offset_B), (void *)beta, (void *)((intptr_t)C + offset_C), (void *)((intptr_t)D + offset_D));
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%lf seconds, %lf FLOPS\n", time_spent, (double)FLO / time_spent);
        imp.TAPP_destroy_tensor_product(plan);
        imp.TAPP_destroy_executor(exec);
        imp.TAPP_destroy_handle(handle);
        free(alpha);
        free(beta);
        free(A);
        free(B);
        free(C);
        free(D);
        for (int i = 0; i < NUMBER_OF_TENSOR_INFOS; i++)
        {
            imp.TAPP_destroy_tensor_info(*tensors[i]);
            free(tensors[i]);
            free(indices[i]);
        }
        free(tensors);
        free(indices);
        
        if (error != 0) return -1;
    }
    unload_strides();
    unload_implementation(&imp);
    return 0;
}
