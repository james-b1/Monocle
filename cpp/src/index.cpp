#include "monocle.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "search.h"

// Opaque struct definition — lives only in this translation unit. Callers
// receive an opaque pointer; the layout is private and free to change.
struct MonocleIndex {
    const float* vectors;   // mmap'd, read-only
    size_t mmap_bytes;      // size passed to munmap
    int n;                  // number of vectors
    int dim;                // dimension per vector
};

extern "C" {

MonocleIndex* monocle_index_load(const char* path, int dim) {
    if (!path || dim <= 0 || dim % 16 != 0) return nullptr;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return nullptr;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return nullptr; }

    const size_t stride = static_cast<size_t>(dim) * sizeof(float);
    if (st.st_size <= 0 ||
        static_cast<size_t>(st.st_size) % stride != 0) {
        close(fd);
        return nullptr;
    }

    void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);  // mmap holds the mapping; fd no longer needed
    if (addr == MAP_FAILED) return nullptr;

    MonocleIndex* idx = new MonocleIndex();
    idx->vectors    = reinterpret_cast<const float*>(addr);
    idx->mmap_bytes = static_cast<size_t>(st.st_size);
    idx->n          = static_cast<int>(st.st_size / stride);
    idx->dim        = dim;
    return idx;
}

void monocle_index_free(MonocleIndex* idx) {
    if (!idx) return;
    if (idx->vectors) {
        munmap(const_cast<float*>(idx->vectors), idx->mmap_bytes);
    }
    delete idx;
}

int monocle_index_size(const MonocleIndex* idx) {
    return idx ? idx->n : 0;
}

int monocle_index_search_topk(
    const MonocleIndex* idx,
    const float* query,
    int k,
    int* out_indices,
    float* out_scores
) {
    if (!idx || !query || !out_indices || !out_scores) return 1;
    if (k <= 0 || k > idx->n) return 1;

    monocle::neon_search_topk(
        idx->vectors, idx->n, idx->dim, query, k, out_indices, out_scores
    );
    return 0;
}

}  // extern "C"
