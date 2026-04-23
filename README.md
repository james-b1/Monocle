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

## Running Phase 4 (complete)

The MCP server exposes Monocle's pipeline as a tool Claude Code can call natively. Two tools are registered:

- `ping(message="hello") -> str` — cheap liveness check, touches no model or index.
- `search_knowledge_base(query, k=5) -> SearchResponse` — runs the Phase 3 agent: `rewrite → search → validate → (retry | END)`. Returns top-k chunks plus the validator's relevance verdict.

`SearchResponse` (the contract Claude sees) is `{query, rewritten_query, is_relevant, reason, attempts, results: [{filename, score, char_offset, char_length, preview}]}`. `is_relevant` is **advisory** — chunks are returned regardless of the validator's verdict, because the validator can be wrong and Claude has more conversational context to judge with.

The agent is constructed **once at server startup** via FastMCP's lifespan (loads ~100 MB of embedder weights + mmaps the index) and held for the server's lifetime. The graph's internal `k` is clamped to `min(20, n_chunks)` so per-call `k` requests never overshoot a small index.

**Failure modes:**

- **Startup**: `build_server` pre-flights the index synchronously. Missing directory, missing `metadata.json`/`vectors.bin`, or 0 chunks → the server refuses to start and prints an actionable message to stderr (e.g., `"Required file missing: data/index/metadata.json. Re-run python -m monocle.ingest <corpus>."`). MCP client sees the disconnect.
- **Per-call**: `search_knowledge_base` always returns a well-formed `SearchResponse`. If Ollama is down, `is_relevant=false`, `results=[]`, `reason` carries Ollama's "please check that Ollama is downloaded, running and accessible" message verbatim. The `ping` tool keeps working — it has no Ollama dependency, so partial outages don't black-hole the whole server.

Prereqs: a Phase-2-built index (`python -m monocle.ingest <dir>`), Ollama running with `llama3.2:3b` pulled (same as Phase 3).

Run the server directly (reads JSON-RPC on stdin, writes on stdout — logs go to stderr):

```bash
PYTHONPATH=python .venv/bin/python -m monocle.mcp \
    --index data/index \
    --root  .              \
    --max-attempts 2
```

Register it with Claude Code so you can call `search_knowledge_base` from any session in this project:

```bash
# adjust the absolute paths to this repo. -e sets env vars; -- separates
# claude flags from the subprocess command. Default scope is `local`
# (private to you in this directory). Use `--scope project` to commit
# the registration to .mcp.json instead.
claude mcp add monocle \
    -e PYTHONPATH=/absolute/path/to/Monocle/python \
    -- /absolute/path/to/Monocle/.venv/bin/python -m monocle.mcp \
       --index /absolute/path/to/Monocle/data/index \
       --root  /absolute/path/to/Monocle

# verify the entry + connection health
claude mcp list             # should show "monocle: ... - ✓ Connected"
claude mcp get monocle      # full config + status

# clean up
claude mcp remove monocle
```

Then in a **new** Claude Code session (MCP servers are loaded at session start; this session won't see the new server until restart):

- `/mcp` should list `monocle` as connected, with `search_knowledge_base` and `ping` as available tools.
- Ask Claude something the corpus actually contains: *"Use the monocle search to find what this project says about ARM Neon SIMD."* Claude should pick `search_knowledge_base` from the tool list, call it, and surface the chunks.
- For a tool-selection sanity check: ask Claude something the corpus *can't* answer (e.g., *"What's the population of Tokyo?"*) — it should NOT call `search_knowledge_base`, since the docstring scopes it to local-document questions only.

Protocol-level smoke test (no Claude Code required) — pipe an MCP handshake + `tools/call` to the server and print responses:

```bash
PYTHONPATH=python .venv/bin/python -c '
import json, subprocess, os
env = os.environ | {"PYTHONPATH": "python"}
p = subprocess.Popen([".venv/bin/python","-m","monocle.mcp","--index","data/index","--root","."],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, env=env)
def send(m): p.stdin.write(json.dumps(m)+"\n"); p.stdin.flush()
def recv(): return json.loads(p.stdout.readline())
send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}); print(recv())
send({"jsonrpc":"2.0","method":"notifications/initialized"})
send({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ping","arguments":{"message":"hi"}}}); print(recv())
send({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_knowledge_base","arguments":{"query":"What is Monocle?","k":2}}}); print(recv())
p.terminate()'
```

Expected: an `initialize` response with `serverInfo.name == "monocle"`, then `pong: hi`, then a `SearchResponse` with `is_relevant=true`, `attempts=1`, and chunks from `README.md`.

## Running Phase 3 (complete)

Phase 3 wires Phases 1 + 2 together under an Ollama-driven state machine:

```
START -> rewrite -> search -> validate -> (relevant ? END : retry up to MAX_ATTEMPTS)
```

Prereqs: Ollama daemon running + `ollama pull llama3.2:3b` + a Phase-2-built index (`python -m monocle.ingest <dir>`). See `Phase 3 prerequisites` below for details.

### CLI

```bash
PYTHONPATH=python .venv/bin/python -m monocle.agent "how does the engine use SIMD?"

# Knobs
PYTHONPATH=python .venv/bin/python -m monocle.agent "<question>" \
    --index data/index \
    --root  .                \
    -k 5                     \
    --max-attempts 2
```

Exits 0 if the validator judged results relevant, 1 otherwise — suitable for scripting.

### Library

```python
from monocle.agent.graph import open_agent

with open_agent("data/index", corpus_root=".") as graph:
    # one-shot:
    final = graph.invoke({"question": "what does ARM Neon do in Phase 1?"})
    print(final["is_relevant"], final["reason"])
    for r in final["results"]:
        print(f"[{r.score:.3f}] {r.filename}")

    # streaming per-node updates (great for tracing):
    for update in graph.stream({"question": "..."}, stream_mode="updates"):
        for node, delta in update.items():
            print(node, delta)
```

`open_agent` is a context manager that owns the index + embedder + LLM client for you. The underlying `build_graph(llm, embedder, index, metadata)` is exposed if you want to wire things manually.

### How retry actually works

On `attempt == 0`, `rewrite` uses the standard prompt at temperature 0.0 (deterministic). If the validator rejects the results, the graph loops back to `rewrite` — but now `state['attempt']` is 1, so the node switches to a **retry-specific system prompt** that includes the previous (failed) rewrite and the validator's failure `reason`, at temperature 0.5. Without this, retry would be pointless (same input → same rewrite → same chunks → same verdict). With it, the second attempt explores genuinely different vocabulary. After `MAX_ATTEMPTS` unsuccessful passes the graph exits honestly with `is_relevant=False`.

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

**Phases 1, 2, 3, and 4 complete.** Monocle is end-to-end: `claude mcp add monocle ...` and the `search_knowledge_base` tool is live in Claude Code, backed by the full Phase 3 LangGraph pipeline running over the Phase 1 SIMD search engine. Ready for Phase 5 (real-document integration testing).

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

### Phase 3 — LangGraph orchestrator

- [x] Step 1: Agent state schema (`monocle.agent.state`) — `AgentState` TypedDict + `SearchResult` frozen dataclass
- [x] Step 2: Query rewriter node (`monocle.agent.nodes.make_rewrite_query`) — Ollama / `llama3.2:3b`; retry-aware (different prompt + higher temp on attempt > 0)
- [x] Step 3: Search node (`make_search_node`) + `open_index` ctx manager — embeds query, calls Phase 1, resolves chunk_ids via `metadata.json`
- [x] Step 4: Result validator node (`make_validate_results`) — Ollama JSON-mode (`format=<JSON Schema>`); optionally loads full chunk text from disk via `corpus_root` for richer judgment
- [x] Step 5: Graph wiring (`monocle.agent.graph.build_graph`, `open_agent`) + CLI (`python -m monocle.agent`) — `StateGraph` with conditional retry edge bounded by `MAX_ATTEMPTS`

**Phase 3 headline:** `python -m monocle.agent "how does the engine use SIMD?"` runs the full `rewrite → search → validate → (retry | END)` loop. Happy-path queries return in ~5 s (one pass, dominated by ~3 Ollama calls × ~1.5 s warm); impossible queries bail after two attempts with `is_relevant=False` + the validator's reason, rather than confidently returning garbage.

#### Phase 3 prerequisites

- **Ollama daemon running** (defaults to `http://localhost:11434`). Install: <https://ollama.com/download>
- **Pull the model**: `ollama pull llama3.2:3b` (~2 GB)
- **Python client**: included in `requirements.txt` (`ollama==0.6.1`)

> **First-call note:** the first chat completion after the daemon starts pays ~5–10 s to load model weights into RAM. Subsequent calls on the same model are ~150–200 ms on M4.

#### Try the rewriter

```python
from monocle.agent import OllamaClient, make_rewrite_query

rewrite = make_rewrite_query(OllamaClient())  # defaults to llama3.2:3b
print(rewrite({"question": "can you please find me info on FFTs"}))
# {'rewritten_query': 'Fast Fourier Transform algorithm documentation', 'attempt': 1}
```

#### Try rewrite + search end-to-end

Requires a Phase 2 index (e.g., `python -m monocle.ingest .` produces `data/index/`).

```python
from monocle.agent import OllamaClient, make_rewrite_query, make_search_node, open_index
from monocle.ingest.embedder import Embedder

with open_index("data/index") as (index, meta):
    embedder = Embedder(model_name=meta["model"])     # must match metadata['model']
    rewrite  = make_rewrite_query(OllamaClient())
    search   = make_search_node(embedder, index, meta, k=5)

    state = {"question": "how does the c++ engine make searches faster?"}
    state |= rewrite(state)
    state |= search(state)

    print(state["rewritten_query"])
    for r in state["results"]:
        print(f"[{r.score:.3f}] {r.filename} @ {r.char_offset}  {r.preview[:60]}")
```

`make_search_node` validates that the embedder's model matches `metadata['model']` at construction time — a mismatch silently returns garbage (vectors live in different latent spaces), so we fail fast.

#### Add the validator (rewrite → search → validate)

```python
from monocle.agent import (
    OllamaClient, make_rewrite_query, make_search_node,
    make_validate_results, open_index,
)
from monocle.ingest.embedder import Embedder

with open_index("data/index") as (index, meta):
    embedder = Embedder(model_name=meta["model"])
    llm      = OllamaClient()
    rewrite  = make_rewrite_query(llm)
    search   = make_search_node(embedder, index, meta, k=5)
    validate = make_validate_results(llm, corpus_root=".")  # loads full chunks for judgment

    state = {"question": "how does the engine use SIMD to speed up search?"}
    state |= rewrite(state)
    state |= search(state)
    state |= validate(state)
    print(state["is_relevant"], "—", state["reason"])
```

Pass `corpus_root` so the validator reads each chunk's full text (via `char_offset` + `char_length`) instead of just the 160-char preview. Without it, the validator can produce false negatives when relevant content sits mid-chunk; with it, the validator pays ~2× more latency (more input tokens) for sharper judgment.

### Phase 4 — MCP server

- [x] Step 1: Package scaffolding (`monocle.mcp`) — FastMCP server + `ping` smoke-test tool; stdio handshake verified with a real MCP client frame sequence
- [x] Step 2: `search_knowledge_base` tool schema — Pydantic `SearchResponse`/`SearchHit` models; `Annotated[T, Field(...)]` parameter descriptions; `k` bounded `[1, 20]` at the schema level; stub handler returns realistic shape
- [x] Step 3: Handler wired to Phase 3 — `build_server(index_dir, ...)` factory; FastMCP lifespan owns the agent; `asyncio.to_thread` bridges sync `graph.invoke` onto an async tool handler; CLI gains `--index`/`--root`/`--max-attempts`; graph_k clamped to `min(20, n_chunks)`
- [x] Step 4: Error handling — `_preflight_index` validates dir/files/n_chunks at startup with actionable messages; per-call `ConnectionError` (Ollama down) and generic `Exception` paths return structured `SearchResponse` with bounded `reason`; never raise empty-text isError
- [x] Step 5: Registered with Claude Code via `claude mcp add monocle ... -- python -m monocle.mcp --index ...` (local scope); `claude mcp list` reports `✓ Connected`; client health probe completes the full MCP `initialize` handshake against our stdio server

**Phase 4 headline:** `claude mcp add monocle ... -- python -m monocle.mcp --index data/index` registers the server; `claude mcp list` reports `monocle: ... - ✓ Connected`. From any Claude Code session in the project, `/mcp` shows `search_knowledge_base` as a callable tool that runs the full `rewrite → search → validate → (retry | END)` agent against the local SIMD-accelerated index. Ollama-down, missing-index, and generic per-call failures all return structured `SearchResponse` payloads instead of opaque errors.
