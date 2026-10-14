#include "../include/multi_tapp/backend.h"
#include "internal.h"

static TAPP_error load_function(void* lib_handle, void** func, const char* func_name, bool crucial, TAPP_error error_code);

TAPP_error Multi_TAPP_create_backend(Multi_TAPP_backend* backend, const char *libpath)
{
    if (backend == NULL)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend pointer");
        return MULTI_TAPP_ERROR_INVALID_ARGUMENT;
    }

    struct mt_backend* backend_ptr = malloc(sizeof(struct mt_backend));
    if (backend_ptr == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate space for a backend object\n");
        return MULTI_TAPP_ERROR_ALLOCATION_FAILED;
    }
    backend_ptr->lib_handle = dlopen(libpath, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!backend_ptr->lib_handle)
    {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        free(backend_ptr);
        return MULTI_TAPP_ERROR_DLOPEN_FAILED;
    }
    dlerror();
    TAPP_error error = 0;
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_start), "TAPP_start", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_shutdown), "TAPP_shutdown", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_create_attr), "TAPP_create_attr", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_destroy_attr), "TAPP_destroy_attr", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_attr_set), "TAPP_attr_set", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_attr_get), "TAPP_attr_get", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_attr_clear), "TAPP_attr_clear", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_check_success), "TAPP_check_success", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_explain_error), "TAPP_explain_error", false, 0);
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_create_executor), "TAPP_create_executor", true, MULTI_TAPP_ERROR_MISSING_CREATE_EXECUTOR);
    if (error != 0) // Check success of loading TAPP_create_executor
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_destroy_executor), "TAPP_destroy_executor", true, MULTI_TAPP_ERROR_MISSING_DESTROY_EXECUTOR);
    if (error != 0) // Check success of loading TAPP_destroy_executor
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_wait_one), "TAPP_wait_one", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_wait_all), "TAPP_wait_all", false, 0);
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_create_tensor_product), "TAPP_create_tensor_product", true, MULTI_TAPP_ERROR_MISSING_CREATE_TENSOR_PRODUCT);
    if (error != 0) // Check success of loading TAPP_create_tensor_product
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_destroy_tensor_product), "TAPP_destroy_tensor_product", true, MULTI_TAPP_ERROR_MISSING_DESTROY_TENSOR_PRODUCT);
    if (error != 0) // Check success of loading TAPP_destroy_tensor_product
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_execute_product), "TAPP_execute_product", true, MULTI_TAPP_ERROR_MISSING_EXECUTE_PRODUCT);
    if (error != 0) // Check success of loading TAPP_execute_product
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_execute_batched_product), "TAPP_execute_batched_product", false, 0);
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_create_tensor_info), "TAPP_create_tensor_info", true, MULTI_TAPP_ERROR_MISSING_CREATE_TENSOR_INFO);
    if (error != 0) // Check success of loading TAPP_create_tensor_info
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_destroy_tensor_info), "TAPP_destroy_tensor_info", true, MULTI_TAPP_ERROR_MISSING_DESTROY_TENSOR_INFO);
    if (error != 0) // Check success of loading TAPP_destroy_tensor_info
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_get_nmodes), "TAPP_get_nmodes", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_set_nmodes), "TAPP_set_nmodes", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_get_extents), "TAPP_get_extents", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_set_extents), "TAPP_set_extents", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_get_strides), "TAPP_get_strides", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_set_strides), "TAPP_set_strides", false, 0);

    *backend = (Multi_TAPP_backend)backend_ptr;

    return 0;
}

TAPP_error Multi_TAPP_destroy_backend(Multi_TAPP_backend backend)
{
    if (backend == 0)
    {
        fprintf(stderr, "WARNING: Tried to destroy backend 0\n");
        return MULTI_TAPP_SUCCESS;
    }

    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->lib_handle)
    {
        dlclose(backend_ptr->lib_handle);
        backend_ptr->lib_handle = NULL;
    }
    free(backend_ptr);

    return 0;
}

TAPP_error Multi_TAPP_start(Multi_TAPP_backend backend)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_start == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_start\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    TAPP_error error = backend_ptr->TAPP_start();

    return multi_tapp_error_from_backend(error);
}

TAPP_error Multi_TAPP_shutdown(Multi_TAPP_backend backend)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_shutdown == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_shutdown\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    TAPP_error error = backend_ptr->TAPP_shutdown();

    return multi_tapp_error_from_backend(error);
}

static TAPP_error load_function(void* lib_handle, void** func, const char* func_name, bool crucial, TAPP_error error_code)
{
    *func = dlsym(lib_handle, func_name);
    const char* error = dlerror();
    if (error != NULL)
    {
        if (crucial)
        {
            fprintf(stderr, "ERROR: dlsym failed to load crucial function %s\n", func_name);
            return error_code;
        }
        fprintf(stderr, "WARNING: dlsym failed to load optional function %s, the implementation might not support this funcion\n", func_name);
        *func = NULL;
    }

    return 0;
}