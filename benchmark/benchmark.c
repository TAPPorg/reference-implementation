#include "benchmark.h"

void load_imlpementation() {
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

void unload_implementation() {
    if (imp.handle) {
        dlclose(imp.handle);
        imp.handle = NULL;
    }
}

int* count_nmodes(int test_id)
{
    int* nmodes = malloc(sizeof(int) * NUMBER_OF_TENSORS);
    for (int i = 0; i < NUMBER_OF_TENSORS; i++)
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

    int64_t* extents[NUMBER_OF_TENSORS] = {
        malloc(sizeof(int64_t) * nmodes[TENSOR_A]),
        malloc(sizeof(int64_t) * nmodes[TENSOR_B]),
        malloc(sizeof(int64_t) * nmodes[TENSOR_D])
    };
    int64_t* strides[NUMBER_OF_TENSORS] = {
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

    TAPP_tensor_info** tensors = malloc(sizeof(TAPP_tensor_info*) * NUMBER_OF_TENSORS);

    for (int i = 0; i < NUMBER_OF_TENSORS; i++)
    {
        tensors[i] = malloc(sizeof(TAPP_tensor_info));
        imp.TAPP_create_tensor_info(tensors[i], TAPP_F32, nmodes[i], extents[i], strides[i]);
        
        free(extents[i]);
        free(strides[i]);
    }

    free(nmodes);

    return tensors;
}

int64_t** create_indices(int test_id, TAPP_tensor_info** tensors)
{
    int64_t** indices = malloc(sizeof(int64_t*) * NUMBER_OF_TENSORS);
    for (int i = 0; i < NUMBER_OF_TENSORS; i++)
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
    return NULL;
}

void* create_tensor_values(TAPP_tensor_info tensor)
{
    int64_t size = 1;
    int64_t* extents = malloc(sizeof(int64_t) * imp.TAPP_get_nmodes(tensor));
    imp.TAPP_get_extents(tensor, extents);
    for (int i = 0; i < imp.TAPP_get_nmodes(tensor); i++)
    {
        size *= extents[i];
    }
    free(extents);
    return create_random_values(size, TAPP_FLOAT);
}

int main(int argc, char const *argv[])
{
    load_imlpementation(&imp);
    for (int test_id = 0; test_id < NUMBER_OF_TESTS; test_id++)
    {
        printf("Test %d, %s: ", test_id + 1, indices_list[test_id]);
        fflush(stdout);
        TAPP_tensor_info** tensors = create_tensor_infos(test_id);
        int64_t** indices = create_indices(test_id, tensors);
        TAPP_handle handle;
        imp.create_handle(&handle);
        TAPP_tensor_product plan;
        TAPP_prectype prec = TAPP_DEFAULT_PREC;
        imp.TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, *tensors[TENSOR_A], indices[TENSOR_A], TAPP_IDENTITY, *tensors[TENSOR_B], indices[TENSOR_B], TAPP_IDENTITY, *tensors[TENSOR_D], indices[TENSOR_D], TAPP_IDENTITY, *tensors[TENSOR_D], indices[TENSOR_D], prec);
        TAPP_executor exec;
        imp.create_executor(&exec);
        TAPP_status status;
        void* alpha = create_random_values(1, TAPP_FLOAT);
        void* beta = create_random_values(1, TAPP_FLOAT);
        void* A = create_tensor_values(*tensors[TENSOR_A]);
        void* B = create_tensor_values(*tensors[TENSOR_B]);
        void* C = create_tensor_values(*tensors[TENSOR_D]);
        void* D = create_tensor_values(*tensors[TENSOR_D]);
        clock_t start = clock();
        TAPP_error error = imp.TAPP_execute_product(plan, exec, &status, (void *)alpha, (void *)A, (void *)B, (void *)beta, (void *)C, (void *)D);
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%f seconds\n", time_spent);
        imp.TAPP_destroy_tensor_product(plan);
        imp.TAPP_destroy_executor(exec);
        imp.TAPP_destroy_handle(handle);
        free(alpha);
        free(beta);
        free(A);
        free(B);
        free(C);
        free(D);
        for (int i = 0; i < NUMBER_OF_TENSORS; i++)
        {
            imp.TAPP_destroy_tensor_info(*tensors[i]);
            free(tensors[i]);
            free(indices[i]);
        }
        free(tensors);
        free(indices);
        
        if (error != 0) return -1;
    }
    unload_implementation(&imp);
    return 0;
}
