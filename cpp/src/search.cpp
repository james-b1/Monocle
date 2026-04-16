#include "search.h"

#include <arm_neon.h>

namespace monocle {

// --- Kernel 1: forced scalar (pragma blocks autovec) -----------------------
void scalar_dot_product_scores(
    const float* __restrict__ vectors, int n, int dim,
    const float* __restrict__ query, float* __restrict__ out
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

// --- Kernel 2: same loop, compiler is free to vectorize --------------------
void autovec_dot_product_scores(
    const float* __restrict__ vectors, int n, int dim,
    const float* __restrict__ query, float* __restrict__ out
) {
    for (int i = 0; i < n; ++i) {
        const float* v = vectors + static_cast<size_t>(i) * dim;
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            sum += v[j] * query[j];
        }
        out[i] = sum;
    }
}

// --- Kernel 3: hand-written Neon, 4 accumulators, 16 floats per iter -------
//
// Why 4 accumulators: breaks the dependency chain on the accumulator,
// letting the M4's multiple FMA pipes issue in parallel every cycle.
// Why 16 floats per iter: 4 accumulators × 4 Neon lanes = 16, and
// dim = 384 is divisible by 16 (no cleanup loop needed).
// vfmaq_f32 is a fused multiply-add — one instruction, one rounding.
void neon_dot_product_scores(
    const float* __restrict__ vectors, int n, int dim,
    const float* __restrict__ query, float* __restrict__ out
) {
    for (int i = 0; i < n; ++i) {
        const float* v = vectors + static_cast<size_t>(i) * dim;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        for (int j = 0; j < dim; j += 16) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(v + j),      vld1q_f32(query + j));
            acc1 = vfmaq_f32(acc1, vld1q_f32(v + j + 4),  vld1q_f32(query + j + 4));
            acc2 = vfmaq_f32(acc2, vld1q_f32(v + j + 8),  vld1q_f32(query + j + 8));
            acc3 = vfmaq_f32(acc3, vld1q_f32(v + j + 12), vld1q_f32(query + j + 12));
        }

        // Horizontal reduce: sum the 4 accumulators, then sum the 4 lanes
        float32x4_t s01 = vaddq_f32(acc0, acc1);
        float32x4_t s23 = vaddq_f32(acc2, acc3);
        float32x4_t sum = vaddq_f32(s01, s23);
        out[i] = vaddvq_f32(sum);
    }
}

}  // namespace monocle
