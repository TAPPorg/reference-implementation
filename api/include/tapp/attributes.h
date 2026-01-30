#ifndef TAPP_ATTRIBUTES_H_
#define TAPP_ATTRIBUTES_H_

#include "util.h"
#include "types.h"

//TODO: predefined attributes? error conditions?

TAPP_EXPORT TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_handle handle, TAPP_key key, void* value);

TAPP_EXPORT TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_handle handle, TAPP_key key, void** value);

TAPP_EXPORT TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_handle handle, TAPP_key key);

#endif /* TAPP_ATTRIBUTES_H_ */
