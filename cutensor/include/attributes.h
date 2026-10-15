#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_ATTRIBUTES_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_ATTRIBUTES_H_

#include <tapp/attributes.h>

#include <cstring>
#include <map>
#include <vector>

typedef std::map<TAPP_key, std::vector<char>> Attributes;

enum {
    TAPP_CUTENSOR_STREAM,
    TAPP_CUTENSOR_DEVICE,
};

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_ATTRIBUTES_H_ */
