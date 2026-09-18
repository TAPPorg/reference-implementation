#ifndef TAPP_ATTRIBUTES_H_
#define TAPP_ATTRIBUTES_H_

#include <stdint.h>

#include "util.h"
#include "error.h"

typedef intptr_t TAPP_attr;
typedef int TAPP_key;

//TODO: predefined attributes? error conditions?

// Create a "bare" attribute object.
// Other TAPP objects (e.g., TAPP_handle, TAPP_tensor_info)
// can have attributes attached to them, but they do not have to.
TAPP_EXPORT TAPP_error TAPP_create_attr(TAPP_attr* attr);

// Note to implementers: object handles should encode the type of object, so
// that e.g. TAPP_attr_destroy() can verify that the handle is actually an attribute object.
TAPP_EXPORT TAPP_error TAPP_destroy_attr(TAPP_attr* attr);

TAPP_EXPORT TAPP_error TAPP_set_attr(TAPP_attr attr, TAPP_key key, size_t size, void* value);

// size is initially set to the size of the buffer pointed to by value.
// If the attribute is found, size is set to the size of the attribute value and
// value is set to point to a buffer containing the attribute value. If the
// attribute is not found, size is set to 0 and value is set to NULL.
// If the attribute is found but the buffer is too small, size is set to the
// size of the attribute value and value is set to NULL.
TAPP_EXPORT TAPP_error TAPP_get_attr(TAPP_attr attr, TAPP_key key, size_t* size, void** value);

// Same as TAPP_set_attr(attr, key, 0, NULL).
TAPP_EXPORT TAPP_error TAPP_clear_attr(TAPP_attr attr, TAPP_key key);

#endif /* TAPP_ATTRIBUTES_H_ */
