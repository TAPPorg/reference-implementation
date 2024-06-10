#include "product.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{
    /*int EXTA[3] = {4, 2, 3};

    int STRA[3] = {1, 4, 8};

    float A[24] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8
    };


    int EXTB[3] = {2, 3, 2};

    int STRB[3] = {1, 2, 6};

    float B[12] = {
        9, 8,
        7, 6,
        5, 4,

        3, 2,
        9, 8,
        7, 6
    };


    int EXTC[2] = {4, 2};

    int STRC[2] = {1, 4};

    float C[8] = {
        1, 2, 3, 4,
        5, 6, 7, 8
    };


    int EXTD[2] = {4, 2};

    int STRD[2] = {1, 4};

    float D[8] = {
        0, 0, 0, 0,
        0, 0, 0, 0
    };*/

    /*int IDXA = 2;
    int EXTA[2] = {4, 3};
    int STRA[2] = {1, 4};
    float A[12] = {
        1, 2, 3, 4,

        5, 6, 7, 8,

        9, 10, 11, 12
    };

    int IDXB = 2;
    int EXTB[2] = {4, 3};
    int STRB[2] = {1, 4};
    float B[12] = {
        1, 2, 3, 4,

        5, 6, 7, 8,

        9, 10, 11, 12
    };

    int IDXC = 2;
    int EXTC[2] = {4, 3};
    int STRC[2] = {1, 4};
    float C[12] = {
        1, 2, 3, 4,

        5, 6, 7, 8,

        9, 10, 11, 12
    };
    
    int IDXD = 2;
    int EXTD[2] = {4, 3};
    int STRD[2] = {1, 4};
    float D[12] = {
        0, 0, 0, 0,

        0, 0, 0, 0,

        0, 0, 0, 0
    };

    float ALPHA = 1;
    float BETA = 0;

    bool FA = false;
    bool FB = false;
    bool FC = false;

    char EINSUM[] = "ij, ij -> ij";*/

    int IDXA = 3;

    int EXTA[3] = {4, 2, 3};

    int STRA[3] = {1, 4, 8};

    float complex A[24] = {
        1 + 8*I, 2 + 7*I, 3 + 6*I, 4 + 5*I,
        5 + 4*I, 6 + 3*I, 7 + 2*I, 8 + 1*I,

        1 + 8*I, 2 + 7*I, 3 + 6*I, 4 + 5*I,
        5 + 4*I, 6 + 3*I, 7 + 2*I, 8 + 1*I,

        1 + 8*I, 2 + 7*I, 3 + 6*I, 4 + 5*I,
        5 + 4*I, 6 + 3*I, 7 + 2*I, 8 + 1*I
    };


    int IDXB = 3;

    int EXTB[3] = {2, 3, 2};

    int STRB[3] = {1, 2, 6};

    float complex B[12] = {
        9 + 2*I, 8 + 3*I,
        7 + 4*I, 6 + 5*I,
        5 + 6*I, 4 + 7*I,

        3 + 8*I, 2 + 9*I,
        9 + 2*I, 8 + 3*I,
        7 + 4*I, 6 + 5*I
    };


    int IDXC = 2;

    int EXTC[2] = {4, 2};

    int STRC[2] = {1, 4};

    float complex C[8] = {
        1 + 1*I, 2 + 2*I, 3 + 3*I, 4 + 4*I,
        5 + 5*I, 6 + 6*I, 7 + 7*I, 8 + 8*I
    };


    int IDXD = 2;

    int EXTD[2] = {4, 2};

    int STRD[2] = {1, 4};

    float complex D[8] = {
        0 + 0*I, 0 + 0*I, 0 + 0*I, 0 + 0*I,
        0 + 0*I, 0 + 0*I, 0 + 0*I, 0 + 0*I
    };

    float complex ALPHA = 1 + 1*I;
    float complex BETA = 0 + 0*I;

    bool FA = false;
    bool FB = false;
    bool FC = false;

    char EINSUM[] = "ijk, jkl -> il";

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    for (int i = 0; i < 8; i++)
    {
        printf("%f + %fi\n", creal(D[i]), cimag(D[i]));
    }
    
    return 0;
}
