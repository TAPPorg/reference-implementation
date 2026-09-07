/*
 * GENERATED FILE - do not edit by hand.
 * Regenerate with generate_product_instantiations.py (see that file's
 * docstring). Included from product.c.
 */
#define T_A float
#define T_B float
#define T_C float
#define T_D float
#define T_ACC float
#include "product_template.c"
#undef T_A
#undef T_B
#undef T_C
#undef T_D
#undef T_ACC
#define T_A double
#define T_B double
#define T_C double
#define T_D double
#define T_ACC double
#include "product_template.c"
#undef T_A
#undef T_B
#undef T_C
#undef T_D
#undef T_ACC
#define T_A scomplex
#define T_B scomplex
#define T_C scomplex
#define T_D scomplex
#define T_ACC scomplex
#include "product_template.c"
#undef T_A
#undef T_B
#undef T_C
#undef T_D
#undef T_ACC
#define T_A dcomplex
#define T_B dcomplex
#define T_C dcomplex
#define T_D dcomplex
#define T_ACC dcomplex
#include "product_template.c"
#undef T_A
#undef T_B
#undef T_C
#undef T_D
#undef T_ACC
#define T_A double
#define T_B dcomplex
#define T_C double
#define T_D dcomplex
#define T_ACC dcomplex
#include "product_template.c"
#undef T_A
#undef T_B
#undef T_C
#undef T_D
#undef T_ACC
#define T_A float
#define T_B float
#define T_C float
#define T_D float
#define T_ACC double
#include "product_template.c"
#undef T_A
#undef T_B
#undef T_C
#undef T_D
#undef T_ACC
#define T_A scomplex
#define T_B scomplex
#define T_C scomplex
#define T_D scomplex
#define T_ACC dcomplex
#include "product_template.c"
#undef T_A
#undef T_B
#undef T_C
#undef T_D
#undef T_ACC

static const TAPP_product_dispatch_entry TAPP_PRODUCT_DISPATCH[] = {
    { TAPP_F32, TAPP_F32, TAPP_F32, TAPP_F32, TAPP_DEFAULT_PREC, TAPP_EXECUTE_PRODUCT_TEMPLATE(float, float, float, float, float) },
    { TAPP_F64, TAPP_F64, TAPP_F64, TAPP_F64, TAPP_DEFAULT_PREC, TAPP_EXECUTE_PRODUCT_TEMPLATE(double, double, double, double, double) },
    { TAPP_C32, TAPP_C32, TAPP_C32, TAPP_C32, TAPP_DEFAULT_PREC, TAPP_EXECUTE_PRODUCT_TEMPLATE(scomplex, scomplex, scomplex, scomplex, scomplex) },
    { TAPP_C64, TAPP_C64, TAPP_C64, TAPP_C64, TAPP_DEFAULT_PREC, TAPP_EXECUTE_PRODUCT_TEMPLATE(dcomplex, dcomplex, dcomplex, dcomplex, dcomplex) },
    { TAPP_F64, TAPP_C64, TAPP_F64, TAPP_C64, TAPP_DEFAULT_PREC, TAPP_EXECUTE_PRODUCT_TEMPLATE(double, dcomplex, double, dcomplex, dcomplex) },
    { TAPP_F32, TAPP_F32, TAPP_F32, TAPP_F32, TAPP_F64F64_ACCUM_F64, TAPP_EXECUTE_PRODUCT_TEMPLATE(float, float, float, float, double) },
    { TAPP_C32, TAPP_C32, TAPP_C32, TAPP_C32, TAPP_F64F64_ACCUM_F64, TAPP_EXECUTE_PRODUCT_TEMPLATE(scomplex, scomplex, scomplex, scomplex, dcomplex) },
};

#define TAPP_PRODUCT_DISPATCH_COUNT (sizeof(TAPP_PRODUCT_DISPATCH) / sizeof(TAPP_PRODUCT_DISPATCH[0]))
