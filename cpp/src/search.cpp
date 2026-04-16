#include "search.h"

#include <arm_neon.h>

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

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

// --- Shared Neon helper: dot product of one vector against query -----------
// 4 accumulators break the dependency chain; 16 floats per inner iteration.
// Requires dim % 16 == 0.
static inline float neon_dot_product_one(
    const float* __restrict__ v, const float* __restrict__ query, int dim
) {
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

    float32x4_t s01 = vaddq_f32(acc0, acc1);
    float32x4_t s23 = vaddq_f32(acc2, acc3);
    return vaddvq_f32(vaddq_f32(s01, s23));
}

// --- Kernel 3: hand-written Neon, full-score output ------------------------
void neon_dot_product_scores(
    const float* __restrict__ vectors, int n, int dim,
    const float* __restrict__ query, float* __restrict__ out
) {
    for (int i = 0; i < n; ++i) {
        out[i] = neon_dot_product_one(vectors + static_cast<size_t>(i) * dim, query, dim);
    }
}

// --- Kernel 4: fused Neon dot product + top-k selection --------------------
//
// Uses a size-k min-heap. The root is the current threshold — the smallest
// score that's still in the top-k. New score > threshold → pop root, push new.
// O(N log k), O(k) extra memory. Beats O(N log N) full sort when k << N.
void neon_search_topk(
    const float* __restrict__ vectors, int n, int dim,
    const float* __restrict__ query, int k,
    int* __restrict__ out_indices, float* __restrict__ out_scores
) {
    using ScoreIdx = std::pair<float, int>;
    // std::greater<> → min-heap (default priority_queue is max-heap)
    std::priority_queue<ScoreIdx, std::vector<ScoreIdx>, std::greater<>> heap;

    for (int i = 0; i < n; ++i) {
        const float* v = vectors + static_cast<size_t>(i) * dim;
        const float score = neon_dot_product_one(v, query, dim);

        if (static_cast<int>(heap.size()) < k) {
            heap.push({score, i});
        } else if (score > heap.top().first) {
            heap.pop();
            heap.push({score, i});
        }
    }

    // Drain heap → out arrays in *descending* score order.
    // Heap yields ascending (min-heap), so fill from rank=k-1 down to 0.
    for (int rank = k - 1; rank >= 0; --rank) {
        out_scores[rank] = heap.top().first;
        out_indices[rank] = heap.top().second;
        heap.pop();
    }
}

}  // namespace monocle
