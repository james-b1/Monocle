#pragma once

// Internal C++ API for search kernels. The three *_dot_product_scores kernels
// compute the full score array (one float per input vector); they exist for
// benchmarking the SIMD speedup story. neon_search_topk is the production
// path — fused dot product + top-k selection in a single pass.

#include <cstddef>

namespace monocle {

// Full-score kernels (all compute the same thing, differ only in how
// the inner loop is expressed). Used for benchmarks.
void scalar_dot_product_scores(
    const float* vectors, int n, int dim,
    const float* query, float* out
);

void autovec_dot_product_scores(
    const float* vectors, int n, int dim,
    const float* query, float* out
);

void neon_dot_product_scores(
    const float* vectors, int n, int dim,
    const float* query, float* out
);

// Fused dot product + top-k selection. Returns the k highest-scoring vector
// indices (and their scores) in descending score order.
// Preconditions: 0 < k <= n, dim % 16 == 0.
// Writes k ints to out_indices and k floats to out_scores (caller-allocated).
void neon_search_topk(
    const float* vectors, int n, int dim,
    const float* query, int k,
    int* out_indices, float* out_scores
);

}  // namespace monocle
