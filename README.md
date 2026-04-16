# Monocle — Local RAG Engine

A semantic search engine for your own documents, built from scratch. Monocle lets you ask natural-language questions over your local notes, PDFs, and code files — with no cloud exposure, no API costs, and a custom C++ core.

## What It Does

Point Monocle at a directory. It chunks your documents, embeds them as vectors, and stores them locally. When you ask a question, a local LLM rewrites and validates your query, a C++ engine searches 50k+ document chunks in under 10ms, and the result surfaces through Claude Code as a native tool.

```
Your Docs (PDFs, .py, .md)
        ↓
[Python Chunker]  — 500-word chunks, 50-word overlap
        ↓
[Embedding Model]  — 384-float vectors (all-MiniLM-L6-v2, runs locally)
        ↓
[.bin Vector Store]  — flat binary on disk
        ↓
[C++ Flat Index]  ←── ctypes FFI from Python
  • ARM Neon SIMD cosine similarity (M4-optimized)
  • Returns top-k chunk indices + scores in <10ms over 50k vectors
        ↓
[LangGraph Orchestrator]
  Node 1: Query Rewriter (Ollama)
  Node 2: C++ Search
  Node 3: Result Validator → fallback branch if needed
        ↓
[MCP Server]  — exposes search_my_knowledge_base tool
        ↓
Claude Code CLI
```

## Why It's Built This Way

| Decision | Choice | Why |
|---|---|---|
| Index type | Flat (brute-force) | Simpler, still fast with SIMD at moderate scale |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, fast, local, no API key |
| C++/Python bridge | `ctypes` | Lowest friction for a `.dylib` |
| Local LLM | Ollama | Runs natively on M4, no cloud dependency |
| Orchestration | LangGraph | Structured state machine for multi-step agent logic |
| Tool exposure | MCP server | Native Claude Code integration |

## Build Phases

- **Phase 1** — C++ vector engine with ARM Neon SIMD (the core)
- **Phase 2** — Python ingestor: crawl, chunk, embed, serialize
- **Phase 3** — LangGraph orchestrator: query rewriting + validation
- **Phase 4** — MCP server: expose as a Claude Code tool
- **Phase 5** — End-to-end testing with real documents

## Environment

- Machine: M4 Mac (Apple Silicon)
- Compiler: `clang++ -O3 -march=native`
- Python: 3.11+
- Key deps: `sentence-transformers`, `langgraph`, `langchain`, `ollama`, `mcp`, `numpy`

## Getting Started

### Prerequisites

- macOS on Apple Silicon (M-series)
- Xcode or Xcode Command Line Tools (provides `clang++` and `make`). Verify: `xcode-select -p`
- Python 3.11+

### First-time setup

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build the C++ library

```bash
make          # builds build/libmonocle.dylib
make clean    # wipes build/
```

## Running the Current Code (Phase 1, through step 5)

```bash
# 1. Sanity check: verify the .dylib loads and Python can call into it
python sanity_check.py

# 2. Generate 50,000 synthetic unit-normalized vectors + a query vector
#    Writes: data/vectors.bin, data/query.bin, data/ground_truth.json
python scripts/generate_synthetic.py
#    Knobs: --n 50000  --dim 384  --k 10  --seed 42  --out data

# 3. Verify the C++ FFI kernel (Neon) matches numpy's ground truth
python scripts/verify_ffi.py

# 4 & 5. Three-way benchmark: scalar vs. compiler-autovec vs. hand-written Neon
make bench
./build/bench
```

Expected outcome of step 3 verifier:
- All 10 top-k indices match numpy exactly
- Max score diff ~1e-7 or smaller (float32 precision — explained in conversation)

Expected outcome of the benchmark (approximate, will vary ~10% run-to-run):

| Kernel | Mean latency | GFLOPs | Speedup vs scalar |
|---|---|---|---|
| scalar (forced) | ~11.4 ms | ~3.3 | 1.00× |
| autovec (compiler) | ~5.9 ms | ~6.6 | ~1.9× |
| **neon (hand-written)** | **~1.0 ms** | **~38.6** | **~11.5×** |

## Status

Currently in **Phase 1** — building the C++ vector search engine.
- [x] Step 1: Project scaffolding + FFI skeleton
- [x] Step 2: Synthetic vector generator + ground truth
- [x] Step 3: Scalar C++ dot product (correctness verified)
- [x] Step 4: Benchmark harness — scalar baseline: ~11.4 ms mean, 3.3 GFLOPs
- [x] Step 5: ARM Neon SIMD — **~1.0 ms mean, 38.6 GFLOPs, 11.5× speedup** (now the production kernel)
- [ ] Step 6: Top-k selection
- [ ] Step 7: `.dylib` + ctypes end-to-end
