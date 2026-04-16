#pragma once

// Internal C++ API for search kernels. Callers outside the library go through
// the extern "C" layer in api.cpp; this header is for internal composition.

#include <cstddef>

namespace monocle {

// Compute the dot product of `query` (length dim) against each row of
// `vectors` (n rows, each of length dim, row-major contiguous).
// Writes n float32 scores into `out`. Pointers must not alias.
void scalar_dot_product_scores(
    const float* vectors,
    int n,
    int dim,
    const float* query,
    float* out
);

}  // namespace monocle
