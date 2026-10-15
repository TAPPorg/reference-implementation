#ifndef TAPP_ELEMENT_OP_H_
#define TAPP_ELEMENT_OP_H_

#include <stddef.h>

/* The operation applied to each element of an operand, shared by products and permutes */
typedef int TAPP_element_op;

enum
{
    TAPP_IDENTITY = 0,
    TAPP_CONJUGATE = 1,
};

#define TAPP_IN_PLACE NULL

#endif /* TAPP_ELEMENT_OP_H_ */
