/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#include "product.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

void calculate_indices(char* EINSUM, int IDXA, int IDXB, int IDXD, char* indices_A, char* indices_B, char* indices_D);
int calculate_contracted_indices(int IDXA, int IDXB, int IDXD, char* indices_A, char* indices_B, char* indices_D, char* indices_contraction);
void calculate_contracted_extents(int contractions, char* indices_contraction, int IDX, char* indices, int* extents, int* extents_contraction);
void compile_strides(int* strides, int IDX, char* indices, int IDXD, char* indices_D, int contractions, char* indices_contraction, int* free_strides, int* contracted_strides);
int calculate_size(int* extents, int IDX);
void increment_coordinates(int* coordinates, int IDX, int* extents);
void zero_array(int* arr, int size);

/*
 * PRODUCT computes D <- ALPHA * A * B + BETA * C
 * IDXA: the number of dimensions of A
 * EXTA: the extent of A in each dimension
 * STRA: the stride of A in each dimension
 * A: the data of A
 * IDXB: the number of dimensions of B
 * EXTB: the extent of B in each dimension
 * STRB: the stride of B in each dimension
 * B: the data of B
 * IDXC: the number of dimensions of C
 * EXTC: the extent of C in each dimension
 * STRC: the stride of C in each dimension
 * C: the data of C
 * IDXD: the number of dimensions of D
 * EXTD: the extent of D in each dimension
 * STRD: the stride of D in each dimension
 * D: the data of D
 * ALPHA: the scalar alpha
 * BETA: the scalar beta
 * FA: whether A is transposed
 * FB: whether B is transposed
 * FC: whether C is transposed
 * EINSUM: the einsum string
 */

int PRODUCT(int IDXA, int* EXTA, int* STRA, float* A,
            int IDXB, int* EXTB, int* STRB, float* B,
            int IDXC, int* EXTC, int* STRC, float* C,
            int IDXD, int* EXTD, int* STRD, float* D,
            float ALPHA, float BETA, bool FA, bool FB, bool FC, char* EINSUM
) {
    char* indices_A = (char*) malloc(IDXA * sizeof(char));
    char* indices_B = (char*) malloc(IDXB * sizeof(char));
    char* indices_D = (char*) malloc(IDXC * sizeof(char));
    calculate_indices(EINSUM, IDXA, IDXB, IDXD, indices_A, indices_B, indices_D);


    int contractions = (IDXA + IDXB - IDXD) / 2;
    char* indices_contraction = (char*) malloc(contractions * sizeof(char));
    contractions = calculate_contracted_indices(IDXA, IDXB, IDXD, indices_A, indices_B, indices_D, indices_contraction);
    indices_contraction = (char*) realloc(indices_contraction, contractions * sizeof(char));

    int* extents_contraction = (int*) malloc(contractions * sizeof(int));
    calculate_contracted_extents(contractions, indices_contraction, IDXA, indices_A, EXTA, extents_contraction);

    int size_contraction = calculate_size(extents_contraction, contractions);


    int* free_strides_A = (int*) malloc(IDXD * sizeof(int));
    int* contracted_strides_A = (int*) malloc(contractions * sizeof(int));
    compile_strides(STRA, IDXA, indices_A, IDXD, indices_D, contractions, indices_contraction, free_strides_A, contracted_strides_A);

    int* free_strides_B = (int*) malloc(IDXD * sizeof(int));
    int* contracted_strides_B = (int*) malloc(contractions * sizeof(int));
    compile_strides(STRB, IDXB, indices_B, IDXD, indices_D, contractions, indices_contraction, free_strides_B, contracted_strides_B);

    int* coordinates_D = (int*) malloc(IDXD * sizeof(int));
    zero_array(coordinates_D, IDXD);
    int* coordinates_contraction = (int*) malloc(contractions * sizeof(int));
    zero_array(coordinates_contraction, contractions);


    int size_D = calculate_size(EXTD, IDXD);

    for (int i = 0; i < size_D; i++) {
        int index_A_free = 0;
        int index_B_free = 0;
        int index_C = 0;
        int index_D = 0;
        for (int j = 0; j < IDXD; j++) {
            index_A_free += coordinates_D[j] * free_strides_A[j];
            index_B_free += coordinates_D[j] * free_strides_B[j];
            index_C += coordinates_D[j] * STRC[j];
            index_D += coordinates_D[j] * STRD[j];
        }
        //D[index_D] = BETA * (FC ? conjf(C[index_C]) : C[index_C]);
        D[index_D] = BETA * C[index_C];
        for (int j = 0; j < size_contraction; j++) {
            int index_A = index_A_free;
            int index_B = index_B_free;
            for (int i = 0; i < contractions; i++)
            {
                index_A += coordinates_contraction[i] * contracted_strides_A[i];
                index_B += coordinates_contraction[i] * contracted_strides_B[i];
            }
            //D[index_D] += ALPHA * (FA ? conjf(A[index_A]) : A[index_A]) * (FB ? conjf(B[index_B]) : B[index_B]);
            D[index_D] += ALPHA * A[index_A] * B[index_B];
            increment_coordinates(coordinates_contraction, contractions, extents_contraction);
        }
        increment_coordinates(coordinates_D, IDXD, EXTD);
    }
    free(indices_A);
    free(indices_B);
    free(indices_D);
    free(indices_contraction);
    free(extents_contraction);
    free(free_strides_A);
    free(contracted_strides_A);
    free(free_strides_B);
    free(contracted_strides_B);
    free(coordinates_D);
    free(coordinates_contraction);
    return 0;
}

void calculate_indices(char* EINSUM, int IDXA, int IDXB, int IDXD, char* indices_A, char* indices_B, char* indices_D) {
    for (int i = 0,skips = 0; EINSUM[i] != '\0'; i++) {
        if (EINSUM[i] == ',' || EINSUM[i] == '-' || EINSUM[i] == '>' || EINSUM[i] == ' ') {
            skips++;
            continue;
        }
        else if (i < IDXA + skips) {
            indices_A[i] = EINSUM[i];
        }
        else if (i < IDXA + IDXB + skips) {
            indices_B[i - IDXA - skips] = EINSUM[i];
        }
        else if (i < IDXA + IDXB + IDXD + skips) {
            indices_D[i - IDXA - IDXB - skips] = EINSUM[i];
        }
    }
}

int calculate_contracted_indices(int IDXA, int IDXB, int IDXD, char* indices_A, char* indices_B, char* indices_D, char* indices_contraction) {
    int k = 0;
    for (int i = 0; i < IDXA; i++) {
        bool index_found_in_B = false;
        bool index_found_in_D = false;
        for (int j = 0; j < IDXB; j++) {
            if (indices_A[i] == indices_B[j]) {
                index_found_in_B = true;
                break;
            }
        }
        for (int j = 0; j < IDXD; j++) {
            if (indices_A[i] == indices_D[j]) {
                index_found_in_D = true;
                break;
            }
        }
        if (index_found_in_B && !index_found_in_D) {
            indices_contraction[k] = indices_A[i];
            k++;
        }
    }
    return k;
}

void calculate_contracted_extents(int contractions, char* indices_contraction, int IDX, char* indices, int* extents, int* extents_contraction) {
    for (int i = 0; i < contractions; i++) {
        for (int j = 0; j < IDX; j++) {
            if (indices_contraction[i] == indices[j]) {
                extents_contraction[i] = extents[j];
            }
        }
    }
}

void compile_strides(int* strides, int IDX, char* indices, int IDXD, char* indices_D, int contractions, char* indices_contraction, int* free_strides, int* contracted_strides) {
    // Calculate strides for free indices
    for (int i = 0; i < IDXD; i++) {
        bool index_found = false;
        for (int j = 0; j < IDX; j++) {
            if (indices_D[i] == indices[j]) {
                free_strides[i] = strides[j];
                index_found = true;
            }
        }
        if (!index_found) {
            free_strides[i] = 0;
        }
    }

    // Calculate strides for contracted indices
    for (int i = 0; i < contractions; i++) {
        for (int j = 0; j < IDX; j++) {
            if (indices_contraction[i] == indices[j]) {
                contracted_strides[i] = strides[j];
            }
        }
    }
}

int calculate_size(int* extents, int IDX) {
    int size = 1;
    for (int i = 0; i < IDX; i++) {
        size *= extents[i];
    }
    return size;
}

void increment_coordinates(int* coordinates, int IDX, int* extents) {
    if (IDX <= 0) {
        return;
    }

    int k = 0;
    do
    {
        coordinates[k] = (coordinates[k] + 1) % extents[k];
        k++;
    } while (coordinates[k - 1] == 0 && k < IDX);
}

bool compare_arrays(int* arr_a, int* arr_b, int size) {
    for (int i = 0; i < size; i++) {
        if (arr_a[i] != arr_b[i]) {
            return false;
        }
    }
    return true;
}

void zero_array(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = 0;
    }
}