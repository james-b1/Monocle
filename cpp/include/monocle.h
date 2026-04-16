#ifndef MONOCLE_H
#define MONOCLE_H

// Public C API for the Monocle vector search engine.
// This is the ABI boundary with Python (via ctypes). Every symbol here has
// extern "C" linkage so names aren't mangled by the C++ compiler.

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a loaded flat index. Allocated by monocle_index_load,
// freed by monocle_index_free. Callers must not inspect the struct.
typedef struct MonocleIndex MonocleIndex;

// Version string; owned by the library, do not free.
const char* monocle_version(void);

// Load a flat float32 vector index by memory-mapping `path`.
// The file must contain n contiguous vectors of `dim` float32 values each,
// row-major, no header. dim must be divisible by 16 (Neon kernel constraint).
// Returns NULL on error (bad path, bad size, bad dim).
MonocleIndex* monocle_index_load(const char* path, int dim);

// Release the index and unmap the underlying file.
// Safe to pass NULL. Do not use the handle after calling this.
void monocle_index_free(MonocleIndex* idx);

// Number of vectors in the index. Returns 0 for NULL.
int monocle_index_size(const MonocleIndex* idx);

// Search the index: dot-product query against all vectors, return top-k.
// Results are in descending score order.
// Writes k int32s to out_indices and k float32s to out_scores
// (both caller-allocated, length k).
//
// Returns 0 on success, nonzero on bad input.
//
// Thread safety: this function may be called concurrently on the same index
// from multiple threads, as long as each call provides its own output buffers.
// The mmap'd data is read-only and the kernel writes only to the caller buffers.
int monocle_index_search_topk(
    const MonocleIndex* idx,
    const float* query,
    int k,
    int* out_indices,
    float* out_scores
);

#ifdef __cplusplus
}
#endif

#endif
