#ifndef HI_TAPP_TENSOR_H_
#define HI_TAPP_TENSOR_H_

#include <stdint.h>

#include "hi_tapp/error.h"
#include "hi_tapp/attributes.h"

// Spaces are intended for block-sparsity.
// A space determines how a tensor index splits in sectors (tiles).
// Each sector has to have its unique vector of labels.
// The labels are primarily used to match corresponding sectors accross two tensors.
// The labels vector can be just one int - can be assigned by default 
// as the index of the sector within a list of all sectors.
// If two indices are contracted, their sectors with equal label vectors have to have equal extents.
// There has to be a 1-1 mapping between label vectors and sectors.
// Typically, the mapping is implemented by storing labels of all nonzero (present) sectors in one 
// int matrix, a row per sector, and then the block of the Tensor's elements simply refers to the row number.
// One can choose to list all sectors in the space, even those unused.
// A space can be reused for multiple tensors, or multiple times within one tensor. In some applications,
// the space is changed pending a tensor operation.
// Typically, sectors that are zero across the tensor are removed.
// Optionally, a product of spaces is also a space. Then additional
// functions might be needed to extract information on subspaces.
// e.g. to find which of sector elements correspond to a given subspace sector.

typedef intptr_t HI_TAPP_space;

HI_TAPP_error HI_TAPP_create_space(HI_TAPP_space* space,
                        int64_t nsectors,
                        const int64_t* extents,
                        int64_t nlabels, // at least 1
                        const HI_TAPP_attr* labels_names_and_extents); // optional, can be NULL

HI_TAPP_error HI_TAPP_destroy_space(HI_TAPP_space space);

HI_TAPP_error HI_TAPP_get_nsectors(HI_TAPP_space space, int64_t* nsectors);

HI_TAPP_error HI_TAPP_set_nsectors(HI_TAPP_space space,
                           int64_t nsectors);

HI_TAPP_error HI_TAPP_get_sector_extents(HI_TAPP_space space, int64_t nsectors,
                      int64_t* extents);

HI_TAPP_error HI_TAPP_set_sector_extents(HI_TAPP_space space,
                            const int64_t* extents);

HI_TAPP_error HI_TAPP_get_all_sector_labels(HI_TAPP_space space, int64_t nlabels,
                      void* labels);

HI_TAPP_error HI_TAPP_set_all_sector_labels(HI_TAPP_space space,
                            const void* labels);

HI_TAPP_error HI_TAPP_get_labels_names_and_extents(HI_TAPP_space space, 
                                int64_t nlabels,
                                HI_TAPP_attr* labels_names_and_extents);

HI_TAPP_error HI_TAPP_set_labels_names_and_extents(HI_TAPP_space space,
                            const HI_TAPP_attr* labels_names_and_extents);

HI_TAPP_error HI_TAPP_labels_to_sector_index(HI_TAPP_space space,
                            const void* labels,
                            int64_t* sector_index); //this could be a user defined lambda function

HI_TAPP_error HI_TAPP_get_num_subsector_elements(HI_TAPP_space space,
                            const void* labels,
                            HI_TAPP_space* subspaces,
                            void* subsectors_labels, //some can be left out
                            int64_t* nelements);

HI_TAPP_error HI_TAPP_get_subsector_coordinates(HI_TAPP_space space, 
                            const void* labels,
                            HI_TAPP_space* subspaces,
                            const void* subsector_labels, //some can be left out
                            int64_t nelements,
                            void* coordinates);

#endif /* HI_TAPP_TENSOR_H_ */
