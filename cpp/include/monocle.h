#ifndef MONOCLE_H
#define MONOCLE_H

// Public C API for the Monocle vector search engine.
// This is the ABI boundary with Python (via ctypes). Every symbol declared
// here must have extern "C" linkage so names aren't mangled by the C++ compiler.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MonocleIndex MonocleIndex;

const char* monocle_version(void);

// Compute dot product of `query` against each row of `vectors`.
// `vectors` is n rows of `dim` float32 values each, row-major contiguous.
// `query` is `dim` float32 values. `out_scores` is n float32 values (caller-allocated).
// Returns 0 on success, nonzero on bad input.
int monocle_dot_product_scores(
    const float* vectors,
    int n,
    int dim,
    const float* query,
    float* out_scores
);

#ifdef __cplusplus
}
#endif

#endif
