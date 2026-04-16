#pragma once

// Internal C++ API for search kernels. All three compute the same thing
// (dot product of `query` against each row of `vectors`, n x dim row-major,
// unit-normalized so dot product == cosine similarity). They differ only in
// how the inner loop is expressed.

#include <cstddef>

namespace monocle {

// Strictly scalar baseline — a #pragma prevents the compiler from
// auto-vectorizing this loop. Used as the correctness reference and the
// "before" number in benchmarks.
void scalar_dot_product_scores(
    const float* vectors, int n, int dim,
    const float* query, float* out
);

// Plain scalar loop, compiled with -O3 -march=native. The compiler's
// auto-vectorizer is free to emit Neon. Measures what the compiler gives
// you for free.
void autovec_dot_product_scores(
    const float* vectors, int n, int dim,
    const float* query, float* out
);

// Hand-written ARM Neon with 4 independent accumulators. Requires
// dim % 16 == 0 (true for all-MiniLM-L6-v2's 384-dim output).
void neon_dot_product_scores(
    const float* vectors, int n, int dim,
    const float* query, float* out
);

}  // namespace monocle
