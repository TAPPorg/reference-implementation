#ifndef TAPP_REF_IMPL_MULTI_TAPP_STATUS_H_
#define TAPP_REF_IMPL_MULTI_TAPP_STATUS_H_

#include "tapp/status.h"
#include "handle.h"

struct Multi_TAPP_status
{
    uint64_t impl_id;
    TAPP_status* status;
};

#endif /* TAPP_REF_IMPL_MULTI_TAPP_STATUS_H_ */