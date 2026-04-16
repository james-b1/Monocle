// Benchmark & correctness check for the three dot-product kernels:
// scalar (forced), autovec (compiler), and neon (hand-written).

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "search.h"

namespace {

struct MappedFloats {
    const float* data;
    size_t bytes;
};

MappedFloats mmap_floats(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { std::perror(path); std::exit(1); }
    struct stat st;
    if (fstat(fd, &st) < 0) { std::perror("fstat"); std::exit(1); }
    void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    close(fd);
    return {reinterpret_cast<const float*>(addr), static_cast<size_t>(st.st_size)};
}

struct Stats {
    double mean, p50, p99, min_v, max_v;
};

Stats compute_stats(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    double sum = 0.0;
    for (double t : times) sum += t;
    return {
        sum / times.size(),
        times[times.size() / 2],
        times[static_cast<size_t>(times.size() * 0.99)],
        times.front(),
        times.back()
    };
}

using Kernel = void (*)(const float*, int, int, const float*, float*);

Stats benchmark_kernel(
    Kernel kernel, const float* vecs, int n, int dim,
    const float* query, float* scratch, int warmup, int iterations
) {
    for (int i = 0; i < warmup; ++i) {
        kernel(vecs, n, dim, query, scratch);
    }
    std::vector<double> times;
    times.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        kernel(vecs, n, dim, query, scratch);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return compute_stats(std::move(times));
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

void print_row(const char* name, const Stats& s, double baseline_mean, int n, int dim) {
    const double flops = 2.0 * n * dim;
    const double gflops = (flops / (s.mean * 1e-3)) / 1e9;
    const double speedup = baseline_mean / s.mean;
    std::printf("  %-10s  %8.3f  %8.3f  %8.3f  %8.2f  %8.2fx\n",
                name, s.mean, s.p50, s.p99, gflops, speedup);
}

}  // namespace

int main() {
    constexpr int DIM = 384;
    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 200;

    auto vec = mmap_floats("data/vectors.bin");
    auto query = mmap_floats("data/query.bin");
    const int n = static_cast<int>(vec.bytes / (DIM * sizeof(float)));

    std::printf("Loaded %d vectors x %d dims (%.1f MB)\n\n",
                n, DIM, vec.bytes / (1024.0 * 1024.0));

    // Correctness: run all three, compare neon & autovec against scalar
    std::vector<float> scalar_scores(n), autovec_scores(n), neon_scores(n);
    monocle::scalar_dot_product_scores(vec.data, n, DIM, query.data, scalar_scores.data());
    monocle::autovec_dot_product_scores(vec.data, n, DIM, query.data, autovec_scores.data());
    monocle::neon_dot_product_scores(vec.data, n, DIM, query.data, neon_scores.data());

    const float autovec_diff = max_abs_diff(scalar_scores, autovec_scores);
    const float neon_diff    = max_abs_diff(scalar_scores, neon_scores);
    std::printf("Correctness (max |scalar - kernel|, threshold 1e-4):\n");
    std::printf("  autovec  vs scalar:  %.2e\n", autovec_diff);
    std::printf("  neon     vs scalar:  %.2e\n\n", neon_diff);
    if (autovec_diff > 1e-4f || neon_diff > 1e-4f) {
        std::printf("CORRECTNESS FAILURE\n");
        return 1;
    }

    // Benchmarks
    std::printf("Benchmark  (%d iterations, %d warmup)\n", ITERATIONS, WARMUP);
    std::printf("  %-10s  %8s  %8s  %8s  %8s  %9s\n",
                "kernel", "mean ms", "p50 ms", "p99 ms", "GFLOPs", "speedup");
    std::printf("  %-10s  %8s  %8s  %8s  %8s  %9s\n",
                "----------", "--------", "--------", "--------", "--------", "---------");

    std::vector<float> scratch(n);

    auto scalar_stats = benchmark_kernel(
        monocle::scalar_dot_product_scores, vec.data, n, DIM, query.data,
        scratch.data(), WARMUP, ITERATIONS);
    print_row("scalar", scalar_stats, scalar_stats.mean, n, DIM);

    auto autovec_stats = benchmark_kernel(
        monocle::autovec_dot_product_scores, vec.data, n, DIM, query.data,
        scratch.data(), WARMUP, ITERATIONS);
    print_row("autovec", autovec_stats, scalar_stats.mean, n, DIM);

    auto neon_stats = benchmark_kernel(
        monocle::neon_dot_product_scores, vec.data, n, DIM, query.data,
        scratch.data(), WARMUP, ITERATIONS);
    print_row("neon", neon_stats, scalar_stats.mean, n, DIM);

    munmap(const_cast<float*>(vec.data), vec.bytes);
    munmap(const_cast<float*>(query.data), query.bytes);
    return 0;
}
