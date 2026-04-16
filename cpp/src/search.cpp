#include "search.h"

namespace monocle {

// Strictly scalar baseline. The `vectorize(disable)` pragma prevents clang's
// auto-vectorizer from turning this into Neon automatically, so the step-5
// hand-written SIMD comparison is honest.
//
// __restrict__ tells the compiler these pointers don't alias, unlocking
// reordering optimizations the compiler would otherwise be too cautious to do.
void scalar_dot_product_scores(
    const float* __restrict__ vectors,
    int n,
    int dim,
    const float* __restrict__ query,
    float* __restrict__ out
) {
    for (int i = 0; i < n; ++i) {
        const float* v = vectors + static_cast<size_t>(i) * dim;
        float sum = 0.0f;
        #pragma clang loop vectorize(disable)
        for (int j = 0; j < dim; ++j) {
            sum += v[j] * query[j];
        }
        out[i] = sum;
    }
}

}  // namespace monocle
