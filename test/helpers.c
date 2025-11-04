/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */

#include "helpers.h"
#include <stdlib.h>
#include <stdio.h>

void print_tensor_s(int nmode, const int64_t *extents, const int64_t *strides, const float *data)
{
    int64_t *coords = malloc(nmode * sizeof(int64_t));
    int64_t size = 1;
    for (size_t i = 0; i < nmode; i++)
    {
        coords[i] = 0;
        size *= extents[i];
    }
    printf("\t");
    for (size_t i = 0; i < size; i++)
    {
        int64_t index = 0;
        for (size_t i = 0; i < nmode; i++)
        {
            index += coords[i] * strides[i];
        }
        printf("%.3f", data[index]);

        if (nmode <= 0)
            continue;

        int k = 0;
        do
        {
            if (k != 0)
            {
                printf("\n");
                if (i < size - 1)
                {
                    printf("\t");
                }
            }
            else
            {
                printf(" ");
            }
            coords[k] = (coords[k] + 1) % extents[k];
            k++;
        } while (coords[k - 1] == 0 && k < nmode);
    }
    free(coords);
}

void print_tensor_c(int nmode, const int64_t *extents, const int64_t *strides, const float complex *data)
{
    int64_t *coords = malloc(nmode * sizeof(int64_t));
    int64_t size = 1;
    for (size_t i = 0; i < nmode; i++)
    {
        coords[i] = 0;
        size *= extents[i];
    }
    printf("\t");
    for (size_t i = 0; i < size; i++)
    {
        int64_t index = 0;
        for (size_t i = 0; i < nmode; i++)
        {
            index += coords[i] * strides[i];
        }
        printf("%.3f+%.3fi", crealf(data[index]), cimagf(data[index]));

        if (nmode <= 0)
            continue;

        int k = 0;
        do
        {
            if (k != 0)
            {
                printf("\n");
                if (i < size - 1)
                {
                    printf("\t");
                }
            }
            else
            {
                printf(" ");
            }
            coords[k] = (coords[k] + 1) % extents[k];
            k++;
        } while (coords[k - 1] == 0 && k < nmode);
    }
    free(coords);
}

void print_tensor_d(int nmode, const int64_t *extents, const int64_t *strides, const double *data)
{
    int64_t *coords = malloc(nmode * sizeof(int64_t));
    int64_t size = 1;
    for (size_t i = 0; i < nmode; i++)
    {
        coords[i] = 0;
        size *= extents[i];
    }
    printf("\t");
    for (size_t i = 0; i < size; i++)
    {
        int64_t index = 0;
        for (size_t i = 0; i < nmode; i++)
        {
            index += coords[i] * strides[i];
        }
        printf("%.3f", data[index]);

        if (nmode <= 0)
            continue;

        int k = 0;
        do
        {
            if (k != 0)
            {
                printf("\n");
                if (i < size - 1)
                {
                    printf("\t");
                }
            }
            else
            {
                printf(" ");
            }
            coords[k] = (coords[k] + 1) % extents[k];
            k++;
        } while (coords[k - 1] == 0 && k < nmode);
    }
    free(coords);
}

void print_tensor_z(int nmode, const int64_t *extents, const int64_t *strides, const double complex *data)
{
    int64_t *coords = malloc(nmode * sizeof(int64_t));
    int64_t size = 1;
    for (size_t i = 0; i < nmode; i++)
    {
        coords[i] = 0;
        size *= extents[i];
    }
    printf("\t");
    for (size_t i = 0; i < size; i++)
    {
        int64_t index = 0;
        for (size_t i = 0; i < nmode; i++)
        {
            index += coords[i] * strides[i];
        }
        printf("%.3f+%.3fi", crealf(data[index]), cimagf(data[index]));

        if (nmode <= 0)
            continue;

        int k = 0;
        do
        {
            if (k != 0)
            {
                printf("\n");
                if (i < size - 1)
                {
                    printf("\t");
                }
            }
            else
            {
                printf(" ");
            }
            coords[k] = (coords[k] + 1) % extents[k];
            k++;
        } while (coords[k - 1] == 0 && k < nmode);
    }
    free(coords);
}
