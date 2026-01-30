#ifndef TAPP_TYPES_H_
#define TAPP_TYPES_H_

#include <stdint.h>

// Temporary fix for circular dependencies (specifically between handle and error)

typedef intptr_t TAPP_attr;
typedef int TAPP_key;
typedef int TAPP_error;
typedef intptr_t TAPP_executor;
typedef intptr_t TAPP_handle;
typedef intptr_t TAPP_tensor_product;
typedef intptr_t TAPP_status;
typedef intptr_t TAPP_tensor_info;

#endif /* TAPP_TYPES_H_ */