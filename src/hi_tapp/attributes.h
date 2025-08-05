#ifndef HI_TAPP_ATTRIBUTES_H_
#define HI_TAPP_ATTRIBUTES_H_

#include <stdint.h>

#include "hi_tapp/error.h"

typedef intptr_t HI_TAPP_attr;
typedef int HI_TAPP_key;

//TODO: predefined attributes? error conditions?

HI_TAPP_error HI_TAPP_attr_set(HI_TAPP_attr attr, HI_TAPP_key key, void* value);

HI_TAPP_error HI_TAPP_attr_get(HI_TAPP_attr attr, HI_TAPP_key key, void** value);

HI_TAPP_error HI_TAPP_attr_clear(HI_TAPP_attr attr, HI_TAPP_key key);

#endif /* HI_TAPP_ATTRIBUTES_H_ */
