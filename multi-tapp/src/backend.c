#include "../include/backend.h"

static TAPP_error load_function(void* lib_handle, void** func, const char* func_name, bool crucial, TAPP_error error_code);

TAPP_error Multi_TAPP_create_backend(TAPP_backend* backend, const char *libpath)
{
    struct Multi_TAPP_backend* backend_ptr = malloc(sizeof(struct Multi_TAPP_backend));
    backend_ptr->lib_handle = dlopen(libpath, RTLD_LAZY);
    if (!backend_ptr->lib_handle)
    {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        free(backend_ptr);
        return 1; // TODO: return error
    }
    dlerror();
    // TODO: Check if each function loaded correctly, if not they will be deemed unimplemented and set to NULL for later error checking.
    TAPP_error error = 0;
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_create_attr), "TAPP_create_attr", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_destroy_attr), "TAPP_destroy_attr", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_attr_set), "TAPP_attr_set", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_attr_get), "TAPP_attr_get", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_attr_clear), "TAPP_attr_clear", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_check_success), "TAPP_check_success", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_explain_error), "TAPP_explain_error", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_create_executor), "TAPP_create_executor", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_destroy_executor), "TAPP_destroy_executor", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_wait_one), "TAPP_wait_one", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_wait_all), "TAPP_wait_all", false, 0);
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_create_tensor_product), "TAPP_create_tensor_product", true, 3);
    if (error != 0) // Check success of loading TAPP_create_tensor_product
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_destroy_tensor_product), "TAPP_destroy_tensor_product", false, 0);
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_execute_product), "TAPP_execute_product", true, 4);
    if (error != 0) // Check success of loading TAPP_execute_product
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_execute_batched_product), "TAPP_execute_batched_product", false, 0);
    error = load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_create_tensor_info), "TAPP_create_tensor_info", true, 5);
    if (error != 0) // Check success of loading TAPP_create_tensor_info
    {
        dlclose(backend_ptr->lib_handle);
        free(backend_ptr);
        return error;
    }
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_destroy_tensor_info), "TAPP_destroy_tensor_info", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_get_nmodes), "TAPP_get_nmodes", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_set_nmodes), "TAPP_set_nmodes", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_get_extents), "TAPP_get_extents", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_set_extents), "TAPP_set_extents", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_get_strides), "TAPP_get_strides", false, 0);
    load_function(backend_ptr->lib_handle, (void**)(&backend_ptr->TAPP_set_strides), "TAPP_set_strides", false, 0);

    *backend = (TAPP_backend)backend_ptr;

    return 0;
}

TAPP_error Multi_TAPP_destroy_backend(TAPP_backend backend)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->lib_handle)
    {
        dlclose(backend_ptr->lib_handle);
        backend_ptr->lib_handle = NULL;
    }
    free(backend_ptr);

    return 0;
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