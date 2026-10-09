#ifndef TAPP_MULTI_TAPP_ERROR_H_
#define TAPP_MULTI_TAPP_ERROR_H_

#include "backend.h"
#include "tapp/error.h"
#include <string.h>

TAPP_EXPORT bool Multi_TAPP_check_success(TAPP_backend backend, TAPP_error error);

TAPP_EXPORT size_t Multi_TAPP_explain_error(TAPP_backend backend,
                                            TAPP_error error,
                                            size_t maxlen,
                                            char* message);

#endif /* TAPP_MULTI_TAPP_ERROR_H_ */