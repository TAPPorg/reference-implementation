#include "../include/util.h"

int64_t compute_index(const int64_t* coordinates, int nmode, const int64_t* strides)
{
    int64_t index = 0;
    for (int i = 0; i < nmode; i++)
    {
        index += coordinates[i] * strides[i];
    }
    return index;

}

void increment_coordinates(int64_t* coordinates, int nmode, const int64_t* extents)
{
    if (nmode <= 0)
    {
        return;
    }

    int k = 0;
    do
    {
        coordinates[k] = (coordinates[k] + 1) % extents[k];
        k++;
    } while (coordinates[k - 1] == 0 && k < nmode);
}

cutensorOperator_t translate_operator(TAPP_element_op op)
{
    switch (op)
    {
    case TAPP_IDENTITY:
        return CUTENSOR_OP_IDENTITY;
        break;
    case TAPP_CONJUGATE:
        return CUTENSOR_OP_CONJ;
        break;
    default: // TODO: Default should probably be an error
        return CUTENSOR_OP_IDENTITY;
        break;
    }
}
