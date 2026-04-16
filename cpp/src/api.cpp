#include "monocle.h"
#include "search.h"

extern "C" {

const char* monocle_version(void) {
    return "Monocle C++ engine v0.0.1";
}

int monocle_dot_product_scores(
    const float* vectors,
    int n,
    int dim,
    const float* query,
    float* out_scores
) {
    if (!vectors || !query || !out_scores || n <= 0 || dim <= 0) {
        return 1;
    }
    monocle::scalar_dot_product_scores(vectors, n, dim, query, out_scores);
    return 0;
}

}  // extern "C"
