#include "../include/handle.h"

char* paths[] = {
    "./libtapp-reference.so",
    "./libtapp-tblis.so"
};

TAPP_error load_function(void* lib_handle, void** func, char* func_name, bool crucial, TAPP_error error_code);

TAPP_error TAPP_create_handle(uint64_t impl_id, TAPP_handle* handle)
{
    struct Multi_TAPP_handle* multi_tapp_handle = malloc(sizeof(struct Multi_TAPP_handle));
    multi_tapp_handle->lib_handle = dlopen(paths[impl_id], RTLD_LAZY);
    if (!multi_tapp_handle->lib_handle)
    {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1; // TODO: return error
    }
    dlerror();
    multi_tapp_handle->impl_id = impl_id;
    // TODO: Check if each function loaded correctly, if not they will be deemed unimplemented and set to NULL for later error checking.
    TAPP_error error = 0;
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_attr_set), "TAPP_attr_set", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_attr_get), "TAPP_attr_get", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_attr_clear), "TAPP_attr_clear", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_check_success), "TAPP_check_success", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_explain_error), "TAPP_explain_error", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_create_executor), "TAPP_create_executor", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_destroy_executor), "TAPP_destroy_executor", false, 0);
    error = load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_create_handle), "TAPP_create_handle", true, 2);
    if (error != 0) // Check success of loading TAPP_create_handle
    {
        dlclose(multi_tapp_handle->lib_handle);
        free(multi_tapp_handle);
        return error;
    }
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_destroy_handle), "TAPP_destroy_handle", false, 0);
    error = load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_create_tensor_product), "TAPP_create_tensor_product", true, 3);
    if (error != 0) // Check success of loading TAPP_create_tensor_product
    {
        dlclose(multi_tapp_handle->lib_handle);
        free(multi_tapp_handle);
        return error;
    }
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_destroy_tensor_product), "TAPP_destroy_tensor_product", false, 0);
    error = load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_execute_product), "TAPP_execute_product", true, 4);
    if (error != 0) // Check success of loading TAPP_execute_product
    {
        dlclose(multi_tapp_handle->lib_handle);
        free(multi_tapp_handle);
        return error;
    }
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_execute_batched_product), "TAPP_execute_batched_product", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_destroy_status), "TAPP_destroy_status", false, 0);
    error = load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_create_tensor_info), "TAPP_create_tensor_info", true, 5);
    if (error != 0) // Check success of loading TAPP_create_tensor_info
    {
        dlclose(multi_tapp_handle->lib_handle);
        free(multi_tapp_handle);
        return error;
    }
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_destroy_tensor_info), "TAPP_destroy_tensor_info", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_get_nmodes), "TAPP_get_nmodes", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_set_nmodes), "TAPP_set_nmodes", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_get_extents), "TAPP_get_extents", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_set_extents), "TAPP_set_extents", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_get_strides), "TAPP_get_strides", false, 0);
    load_function(multi_tapp_handle->lib_handle, (void**)(&multi_tapp_handle->TAPP_set_strides), "TAPP_set_strides", false, 0);

    *handle = (TAPP_handle)multi_tapp_handle;
    if (multi_tapp_handle->TAPP_create_handle == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_handle\n");
        return 6; // TODO: Return error for non implemented function
    }

    return multi_tapp_handle->TAPP_create_handle(impl_id, &multi_tapp_handle->tapp_handle);
}

TAPP_error TAPP_destroy_handle(TAPP_handle handle)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_destroy_handle == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_handle\n");
        return 6; // TODO: Return error for non implemented function
    }
    TAPP_error error = multi_tapp_handle->TAPP_destroy_handle(multi_tapp_handle->tapp_handle);
    if (multi_tapp_handle->lib_handle)
    {
        dlclose(multi_tapp_handle->lib_handle);
        multi_tapp_handle->lib_handle = NULL;
    }
    free(multi_tapp_handle);

    return error;
}

TAPP_error load_function(void* lib_handle, void** func, char* func_name, bool crucial, TAPP_error error_code)
{
    *func = dlsym(lib_handle, func_name);
    const char* error = dlerror();
    if (error != NULL)
    {
        if (crucial)
        {
            dlclose(lib_handle);
            fprintf(stderr, "ERROR: dlsym failed to load crucial function %s\n", func_name);
            return error_code;
        }
        fprintf(stderr, "WARNING: dlsym failed to load optional function %s, the implementation might not support this funcion\n", func_name);
        *func = NULL;
    }

    return 0;
}