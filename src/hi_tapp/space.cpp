/*
 * Jan Brandejs
 * Universite de Toulouse - August 2025
 */
#include "hi_tapp/tapp_ex_imp.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

struct VectorHasher {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t hash = vec.size();
        for (int i : vec) {
            hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

HI_TAPP_error HI_TAPP_create_space(HI_TAPP_space* space,
                        int64_t nsectors,
                        const int64_t* extents,
                        int64_t nlabels, // at least 1
                        const HI_TAPP_attr* labels_names_and_extents) {

    struct space* space_ptr = (struct space*) malloc(sizeof(struct space));

    space_ptr->nsectors = nsectors;
    space_ptr->nlabels = nlabels;

    space_ptr->extents = (int64_t*)malloc(nsectors * sizeof(int64_t));
    memcpy(space_ptr->extents, extents, nsectors * sizeof(int64_t));

    space_ptr->labels_names_and_extents = (HI_TAPP_attr*)malloc(nsectors * nlabels * sizeof(int64_t));
    memcpy(space_ptr->labels_names_and_extents, labels_names_and_extents, nsectors * nlabels * sizeof(int64_t));

    *space = (HI_TAPP_space)space_ptr;
    

    return 0;
}