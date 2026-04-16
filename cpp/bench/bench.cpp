// Benchmark the scalar dot product kernel against data/vectors.bin + data/query.bin.
// Reports mean, median (p50), and p99 latency over many iterations after warm-up.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "search.h"

namespace {

struct MappedFloats {
    const float* data;
    size_t bytes;
};

// mmap a file read-only. Exits on error.
MappedFloats mmap_floats(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { std::perror(path); std::exit(1); }

    struct stat st;
    if (fstat(fd, &st) < 0) { std::perror("fstat"); std::exit(1); }

    void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { std::perror("mmap"); std::exit(1); }

    close(fd);  // mmap holds the mapping; fd no longer needed
    return {reinterpret_cast<const float*>(addr), static_cast<size_t>(st.st_size)};
}

}  // namespace

int main() {
    constexpr int DIM = 384;
    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 200;

    auto vec = mmap_floats("data/vectors.bin");
    auto query = mmap_floats("data/query.bin");
    const int n = static_cast<int>(vec.bytes / (DIM * sizeof(float)));

    std::printf("Loaded %d vectors x %d dims (%.1f MB)\n",
                n, DIM, vec.bytes / (1024.0 * 1024.0));

    std::vector<float> scores(n);

    // Warm-up: pull data into cache, let the OS stabilize
    for (int i = 0; i < WARMUP; ++i) {
        monocle::scalar_dot_product_scores(vec.data, n, DIM, query.data, scores.data());
    }

    // Measure
    std::vector<double> times_ms;
    times_ms.reserve(ITERATIONS);
    for (int i = 0; i < ITERATIONS; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        monocle::scalar_dot_product_scores(vec.data, n, DIM, query.data, scores.data());
        auto t1 = std::chrono::steady_clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Force the compiler to keep the computation (print a score byte)
    std::printf("sanity: scores[0]=%.6f scores[%d]=%.6f\n",
                scores[0], n - 1, scores[n - 1]);

    // Stats (sort for percentiles)
    std::sort(times_ms.begin(), times_ms.end());
    double sum = 0.0;
    for (double t : times_ms) sum += t;
    const double mean = sum / times_ms.size();
    const double p50 = times_ms[times_ms.size() / 2];
    const double p99 = times_ms[static_cast<size_t>(times_ms.size() * 0.99)];

    std::printf("\nScalar benchmark  (%d iterations after %d warmup)\n",
                ITERATIONS, WARMUP);
    std::printf("  mean:  %.3f ms\n", mean);
    std::printf("  p50:   %.3f ms\n", p50);
    std::printf("  p99:   %.3f ms\n", p99);
    std::printf("  min:   %.3f ms    max: %.3f ms\n",
                times_ms.front(), times_ms.back());

    // Throughput: 2 flops (1 mul + 1 add) per (vector, dim) pair
    const double flops_per_query = 2.0 * n * DIM;
    const double gflops = (flops_per_query / (mean * 1e-3)) / 1e9;
    const double qps = 1000.0 / mean;
    std::printf("  throughput: %.2f GFLOPs   (%.0f queries/sec, single core)\n",
                gflops, qps);

    munmap(const_cast<float*>(vec.data), vec.bytes);
    munmap(const_cast<float*>(query.data), query.bytes);
    return 0;
}
