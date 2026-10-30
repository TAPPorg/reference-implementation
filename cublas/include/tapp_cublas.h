#ifndef TAPP_CUBLAS_H_
#define TAPP_CUBLAS_H_

/*
 * Public definitions of the cuBLAS/TTGT implementation of TAPP, beyond the common interface in tapp.h. They are
 * specific to this implementation: other implementations do not know them.
 */

#include <tapp.h>

enum
{
    TAPP_CUBLAS_EMULATION_STRATEGY,
    TAPP_CUBLAS_EMULATION_MANTISSA_CONTROL,
    TAPP_CUBLAS_EMULATION_NUM_GEMM,
};

/*
 * Additional TAPP_prectype values for this implementation.
 */
enum
{
    /*
     * Variable-precision compute types for fixed-point / Ozaki-style FP64
     * emulation (e.g. cuBLAS emulated fixed-point, or a dedicated Ozaki
     * back-end). The integer in the name is the number of correct decimal
     * digits of precision requested in the accumulation. The TAPP_F_* values
     * are for real outputs, TAPP_C_* for complex. They cover 3..16 digits plus
     * a 34-digit (quad-like) point. The values are encoded so the digit count
     * is recoverable as (value % 1000): real are 1000 + digits, complex are
     * 2000 + digits.
     */
    TAPP_F_3_DIGITS  = 1003,
    TAPP_F_4_DIGITS  = 1004,
    TAPP_F_5_DIGITS  = 1005,
    TAPP_F_6_DIGITS  = 1006,
    TAPP_F_7_DIGITS  = 1007,
    TAPP_F_8_DIGITS  = 1008,
    TAPP_F_9_DIGITS  = 1009,
    TAPP_F_10_DIGITS = 1010,
    TAPP_F_11_DIGITS = 1011,
    TAPP_F_12_DIGITS = 1012,
    TAPP_F_13_DIGITS = 1013,
    TAPP_F_14_DIGITS = 1014,
    TAPP_F_15_DIGITS = 1015,
    TAPP_F_16_DIGITS = 1016,
    TAPP_F_34_DIGITS = 1034,

    TAPP_C_3_DIGITS  = 2003,
    TAPP_C_4_DIGITS  = 2004,
    TAPP_C_5_DIGITS  = 2005,
    TAPP_C_6_DIGITS  = 2006,
    TAPP_C_7_DIGITS  = 2007,
    TAPP_C_8_DIGITS  = 2008,
    TAPP_C_9_DIGITS  = 2009,
    TAPP_C_10_DIGITS = 2010,
    TAPP_C_11_DIGITS = 2011,
    TAPP_C_12_DIGITS = 2012,
    TAPP_C_13_DIGITS = 2013,
    TAPP_C_14_DIGITS = 2014,
    TAPP_C_15_DIGITS = 2015,
    TAPP_C_16_DIGITS = 2016,
    TAPP_C_20_DIGITS = 2020,
    TAPP_C_34_DIGITS = 2034,
};

/* Types used for Ozaki-I scheme:
 * - cuBLAS emulation strategy: eager vs performant
 * - cuBLAS emulation mantissa: dynamic vs fixed
 */
typedef int TAPP_emulation_strategy;

enum
{
    TAPP_EMULATION_STRATEGY_PERFORMANT,
    TAPP_EMULATION_STRATEGY_EAGER,
    TAPP_DEFAULT_EMULATION_STRATEGY = TAPP_EMULATION_STRATEGY_PERFORMANT,
};

typedef int TAPP_emulation_mantissa_control;

enum
{
    TAPP_EMULATION_MANTISSA_CONTROL_DYNAMIC,
    TAPP_EMULATION_MANTISSA_CONTROL_FIXED,
    TAPP_DEFAULT_EMULATION_MANTISSA_CONTROL = TAPP_EMULATION_MANTISSA_CONTROL_DYNAMIC,
};

/* Type used for Ozaki-II scheme. It dictates the number of splits, which is also equal to the number of executed GEMMs.
 * There are not clear relation between the number of splits and the accuracy of the GEMM.
 */
typedef int TAPP_num_gemm;

enum
{
    TAPP_DEFAULT_NUM_GEMM = 10,
    TAPP_2_GEMM = 2,
    TAPP_3_GEMM = 3,
    TAPP_4_GEMM = 4,
    TAPP_5_GEMM = 5,
    TAPP_6_GEMM = 6,
    TAPP_7_GEMM = 7,
    TAPP_8_GEMM = 8,
    TAPP_9_GEMM = 9,
    TAPP_10_GEMM = 10,
    TAPP_11_GEMM = 11,
    TAPP_12_GEMM = 12,
    TAPP_13_GEMM = 13,
    TAPP_14_GEMM = 14,
    TAPP_15_GEMM = 15,
    TAPP_16_GEMM = 16,
    TAPP_17_GEMM = 17,
    TAPP_18_GEMM = 18,
    TAPP_19_GEMM = 19,
    TAPP_20_GEMM = 20,
};

#endif  /* TAPP_CUBLAS_H_ */
