#ifndef TAPP_ATTRIBUTES_H_
#define TAPP_ATTRIBUTES_H_

#include <stdint.h>

#include "util.h"
#include "error.h"

typedef intptr_t TAPP_attr;
typedef int TAPP_key;

// create/destroy an attribute set (backend-defined implementation)
TAPP_EXPORT TAPP_error TAPP_create_attr(TAPP_attr *attr);

TAPP_EXPORT TAPP_error TAPP_destroy_attr(TAPP_attr attr);

// set a key to a value; value is a block of size bytes copied by the backend
TAPP_EXPORT TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, size_t size, void* value);

// get the value for a key (NULL if key not set); output in size and value (value is not copied)
TAPP_EXPORT TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, size_t *size, void** value);

// remove key/value pair
TAPP_EXPORT TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_key key);

#endif /* TAPP_ATTRIBUTES_H_ */
