#ifndef TAPP_REF_IMPL_MULTI_TAPP_EXECUTOR_H_
#define TAPP_REF_IMPL_MULTI_TAPP_EXECUTOR_H_

#include "tapp/executor.h"
#include "handle.h"
#include <stdio.h>

struct Multi_TAPP_executor
{
    uint64_t impl_id;
    TAPP_executor* exec;
};

#endif /* TAPP_REF_IMPL_MULTI_TAPP_EXECUTOR_H_ */