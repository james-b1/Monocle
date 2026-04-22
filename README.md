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

## Running Phase 1 (complete)

```bash
# 1. Generate 50,000 synthetic unit-normalized vectors + a query vector
#    Writes: data/vectors.bin, data/query.bin, data/ground_truth.json
python scripts/generate_synthetic.py
#    Knobs: --n 50000  --dim 384  --k 10  --seed 42  --out data

# 2. End-to-end smoke test: load, search, print top-5
python sanity_check.py

# 3. Verify the production API matches numpy ground truth
python scripts/verify_ffi.py

# 4. Benchmark all four kernels + correctness check of top-k vs full-sort
make bench
./build/bench
```

### Using the Python API

```python
from monocle import ffi
import numpy as np

query = np.fromfile("data/query.bin", dtype=np.float32)

with ffi.Index("data/vectors.bin", dim=384) as idx:
    indices, scores = idx.search(query, k=10)
# indices: np.int32 array of length k, descending score order
# scores:  np.float32 array of length k
```

The `Index` is mmap-backed; load is instant. Safe to share across threads for concurrent reads.

### Expected outcome

Verifier (step 3):
- All 10 top-k indices match numpy exactly
- Max score diff ~3e-8 or smaller (float32 precision)

Benchmark (approximate, varies ~10% run-to-run):

| Kernel | Mean latency | GFLOPs | Speedup vs scalar |
|---|---|---|---|
| scalar (forced) | ~11.4 ms | ~3.3 | 1.00× |
| autovec (compiler) | ~5.9 ms | ~6.6 | ~1.9× |
| neon (full scores) | ~1.0 ms | ~38.6 | ~11.5× |
| **neon + top-k (fused, production)** | **~1.0 ms** | **~38** | **~11.5×** |

## Running Phase 2 (complete)

Ingest a directory of `.md` / `.txt` files into a Phase-1-loadable index:

```bash
# Default output: data/index/{vectors.bin, metadata.json}
PYTHONPATH=python .venv/bin/python -m monocle.ingest <directory>

# Knobs
PYTHONPATH=python .venv/bin/python -m monocle.ingest <directory> \
    --out data/index \
    --model all-MiniLM-L6-v2 \
    --chunk-size 500 \
    --overlap 50 \
    --batch-size 32 \
    --no-progress
```

Search the resulting index, resolving chunk indices back to filenames:

```python
import json, numpy as np
from monocle import ffi
from monocle.ingest import Embedder

meta = json.load(open("data/index/metadata.json"))
emb = Embedder(model_name=meta["model"])

q = emb.encode(["how does SIMD speed up cosine similarity?"])[0]
q /= np.linalg.norm(q)

with ffi.Index("data/index/vectors.bin", dim=meta["dim"]) as idx:
    indices, scores = idx.search(q.astype(np.float32), k=5)

for i, s in zip(indices, scores):
    c = meta["chunks"][i]
    print(f"[{s:.3f}] {c['filename']} @ char {c['char_offset']}")
    print(f"        {c['preview']}")
```

The library API is also callable directly (Phase 4 will use this from the MCP server):

```python
from monocle.ingest import ingest
ingest(root=".", out_dir="data/index")
```

## Status

**Phases 1 and 2 complete; Phase 3 in progress.** Step 1 done — agent state schema (`monocle.agent.state`).

### Phase 1 — C++ vector engine

- [x] Step 1: Project scaffolding + FFI skeleton
- [x] Step 2: Synthetic vector generator + ground truth
- [x] Step 3: Scalar C++ dot product (correctness verified against numpy)
- [x] Step 4: Benchmark harness — scalar baseline: ~11.4 ms, 3.3 GFLOPs
- [x] Step 5: ARM Neon SIMD — ~1.0 ms, 38.9 GFLOPs, 11.6× speedup over scalar
- [x] Step 6: Fused top-k via O(N log k) min-heap (zero intermediate array)
- [x] Step 7: Opaque `Index` handle, mmap-backed load, thread-safe concurrent search

**Phase 1 headline:** 0.98 ms top-10 search over 50,000 × 384-dim vectors on M4, reproducible via `./build/bench`.

### Phase 2 — Python ingestor

- [x] Step 1: File crawler (`monocle.ingest.crawl`) — `.md`/`.txt`, skips hidden, sorted output
- [x] Step 2: Text chunker (`monocle.ingest.chunk_text`) — 500-word windows, 50-word overlap, char offsets preserved
- [x] Step 3: Embedding generator (`monocle.ingest.Embedder`) — `all-MiniLM-L6-v2`, 384-dim float32, MPS-accelerated on M4
- [x] Step 4: Serializer (`monocle.ingest.write_index`) — L2-normalize + atomic write; round-trips through `ffi.Index`
- [x] Step 5: `metadata.json` sidecar + `ingest` CLI (`python -m monocle.ingest <dir>`)

**Phase 2 headline:** end-to-end natural-language search over local docs in one command. `python -m monocle.ingest .` crawls + chunks + embeds + serializes in ~6 seconds for the project's own docs; query results resolve to `(filename, char_offset, preview)` via `metadata.json`.

> **First-run note:** the first `Embedder()` (or `python -m monocle.ingest`) call downloads ~80 MB of model weights from HuggingFace and caches them at `~/.cache/huggingface/hub/`. Subsequent runs are fully offline.

### Phase 3 — LangGraph orchestrator (in progress)

- [x] Step 1: Agent state schema (`monocle.agent.state`) — `AgentState` TypedDict + `SearchResult` frozen dataclass
- [ ] Step 2: Query rewriter node (Ollama, `llama3.2:3b`)
- [ ] Step 3: Search node (wraps Phase 1 `ffi.Index` + resolves `metadata.json`)
- [ ] Step 4: Result validator node (Ollama JSON mode)
- [ ] Step 5: Wire the graph + conditional fallback edges
