# CLAUDE.md — Monocle Project

## Who You're Working With

**James** is a second-year CSE student at Ohio State (IBE Honors Program).

**Background:**
- C/C++: Hands-on through Systems I — cache optimization, SIMD/parallelism, registers, branch optimization. Strong foundational intuition, still learning.
- Python: Comfortable, used in coursework and projects.
- Systems thinking: Strong. Understands memory hierarchy, benchmarking, empirical tuning.
- No prior experience with: LangGraph, MCP, `ctypes`/FFI, embedding models, or RAG architectures.

**How to teach:**
- Why → What → How. Always in that order.
- Connect new concepts to what he already knows. (e.g., cosine similarity ↔ dot products; SIMD lanes ↔ the parallelism labs from Systems I; contiguous float arrays ↔ cache line locality)
- Name the trade-offs at every decision point — he wants to understand the choice, not just copy the answer.
- Don't skip the conceptual layer. Don't dumb it down either. He's capable; he just needs grounding.
- Be direct and concise.

---

## Project: Monocle — Local RAG Engine

A semantic search system for local documents, exposed via MCP so Claude Code can use it as a tool.

Architecture:
```
Docs → Python Chunker → Embedding Model → .bin Vector Store
→ C++ Flat Index (SIMD cosine similarity, ctypes FFI)
→ LangGraph Orchestrator (Ollama: query rewrite + validation)
→ MCP Server → Claude Code
```

---

## Build Phases

### Phase 1 — C++ Vector Engine (current)
Goal: A `.dylib` Python can call to do fast cosine similarity search over a flat vector store.

1. Data format: contiguous arrays of 384 `float32` values in a binary file
2. Pure C++ cosine similarity — verify correctness first, no SIMD yet
3. ARM Neon SIMD optimization (`<arm_neon.h>`) — connect to parallelism labs
4. `search()` function: load flat index, return top-k indices + scores
5. Compile: `clang++ -O3 -march=native -shared -fPIC`
6. Benchmark: target sub-10ms over 50,000 vectors

Teaching moments for Phase 1:
- Why cosine similarity (not Euclidean) for semantic vectors
- How SIMD lanes map to "N ops per cycle" from Systems I
- Memory layout: why contiguous `float[]` matters for cache (he knows this)
- FFI basics: how Python `ctypes` loads a `.dylib` and passes pointers

### Phase 2 — Python Ingestor
1. File crawling (`pathlib`, handle PDF/markdown/Python)
2. Text chunking (500-word chunks, 50-word overlap — explain why overlap)
3. Embedding generation (`sentence-transformers`)
4. Binary serialization (`numpy`)
5. Metadata JSON sidecar: chunk_id → {filename, char_offset, preview}

### Phase 3 — LangGraph Orchestrator
1. State schema (TypedDict)
2. Node 1: Query rewriter (Ollama)
3. Node 2: C++ search via ctypes
4. Node 3: Validator (asks Ollama: "are these results relevant?")
5. Conditional edges: validator says no → fallback branch
6. Wire the graph

### Phase 4 — MCP Server
1. Install `mcp` Python SDK
2. Define tool schema: `search_my_knowledge_base`
3. Handler calls LangGraph pipeline
4. Expose via stdio transport
5. Register in Claude Code config

### Phase 5 — Integration & Real-World Testing
End-to-end test: "Search my Digital Logic notes for the truth table for a JK Flip-Flop."

---

## Ground Rules

1. **Start and stay in phase order.** Don't skip ahead.
2. **Before writing code in a new phase**, explain the concept, connect to prior knowledge, surface key design decisions.
3. **At every decision point**, name the alternatives and explain why you're choosing this path.
4. **Benchmark as you go**, especially Phase 1. Write the benchmark harness before optimizing.
5. **Flag memory safety issues** at the C++/Python FFI boundary. That's where subtle bugs live.
6. **Keep scope tight.** Extensions (HNSW, web fallback, etc.) are Phase 6+.

## Learning Companion Rule

**After every code block or implementation step, invoke `/learn` to provide relevant documentation and reading material.** This is not optional — it is part of every implementation response.

The goal is that James can always go read where something comes from. Every concept introduced, every library used, every API called should have a pointer to the authoritative source.

---

## Environment

- Machine: M4 Mac (Apple Silicon — ARM architecture)
- Target compiler: `clang++ -O3 -march=native`
- Python: 3.11+
- Key deps to install when ready: `sentence-transformers`, `langgraph`, `langchain`, `ollama`, `mcp`, `numpy`
