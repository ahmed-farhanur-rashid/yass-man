# YASS-MAN Architecture

## Overview

YASS-MAN is a multi-stage search pipeline that wraps SearXNG to provide:
- **Better recall** — via query expansion (one query becomes up to 5)
- **Better precision** — via ML reranking (cross-encoder scores every result)
- **Near-duplicate removal** — via embedding clustering
- **Synthesized answers** — via a local GGUF LLM with inline citations

Everything runs on your machine. No external APIs, no telemetry.

---

## Pipeline Data Flow

```
User query (string)
       │
       ▼
┌─────────────┐   RouterResult
│   Router    │──────────────────────────────────────────────────┐
└─────────────┘   intent, num_expansions, strategy               │
       │                                                         │
       ▼                                                         │
┌─────────────┐   list[str]  (original + up to 4 variants)      │
│   Expander  │                                                  │
└─────────────┘                                                  │
       │                                                         │
       ▼                                                         │
┌─────────────┐   list[RawResult]  (all queries, parallel)       │
│  Retriever  │──── asyncio.gather → SearXNG /search             │
└─────────────┘                                                  │
       │                                                         │
       ▼                                                         │
┌─────────────┐   list[AggregatedResult]                         │
│  Aggregator │──── URL-normalize → dedup → merge snippets       │
└─────────────┘                                                  │
       │                                                         │
       ▼                                                         │
┌─────────────┐   list[AggregatedResult]  (smaller)              │
│   Embedder  │──── embed → cosine sim → Union-Find cluster      │
└─────────────┘                                                  │
       │                                                         │
       ▼                                                         │
┌─────────────┐   list[RankedResult]  (top-K, sorted)            │
│   Reranker  │──── cross-encoder (query, doc) pairs             │
└─────────────┘                                                  │
       │                                                         │
       ▼                                                         │
┌─────────────┐   SynthesisResult  (answer + citations)          │
│ Synthesizer │──── local GGUF LLM                               │
└─────────────┘     └─ None if disabled or fails                 │
       │                                                         │
       ▼                                                         │
┌─────────────┐   SearchResponse (JSON)                          │
│   Logger    │──── JSONL per query                              │
└─────────────┘                                                  │
       │                                                         │
       └─────────────────────────────────────────────────────────┘
               original RouterResult passed to Synthesizer
```

---

## Module Reference

### `backend/config.py`

**Class:** `Settings` (Pydantic `BaseSettings`)

Loaded once from `.env` via `get_settings()` (cached singleton). Holds:
- `SEARXNG_URL` — which SearXNG instance to query
- `MODEL_DIR`, `LOG_DIR` — filesystem paths
- `ENABLE_LLM`, `ENABLE_CLUSTERING`, `ENABLE_FEEDBACK` — feature flags
- `TOP_K_RESULTS`, `MAX_EXPANDED_QUERIES`, `SEARCH_TIMEOUT_SECONDS` — pipeline tuning

**Rule:** Never hardcode URLs or paths. Always read from `Settings`.

---

### `backend/model_config_loader.py`

**Class:** `ModelConfig` (dataclass tree)

Loaded once from `model_config.yaml` via `load_model_config()` (cached singleton). Sub-configs:

| Attribute | Type | Purpose |
|---|---|---|
| `embedder` | `EmbedderConfig` | HuggingFace repo, batch size, seq length |
| `reranker` | `RerankerConfig` | HuggingFace repo, top-K, seq length |
| `router` | `RouterConfig` | `rule-based` or `model` mode |
| `llm` | `LLMConfig` | GGUF path, context length, temperature, GPU layers |
| `pipeline` | `PipelineConfig` | dedup threshold, max expansions, timeout |

**Rule:** Never hardcode model names anywhere else. Always read from `ModelConfig`.

---

### `backend/pipeline/router.py`

**Class:** `QueryRouter`

Classifies queries into 5 intent types using regex patterns (no ML model required):

| Intent | Trigger | Expansions | Strategy |
|---|---|---|---|
| `fact` | "what is", "who is", short queries | 2 | paraphrase |
| `compare` | "vs", "versus", "difference between" | 4 | comparison |
| `troubleshoot` | "error", "not working", "how to fix" | 3 | community |
| `research` | "best", "how to", "tutorial", "guide" | 5 | technical |
| `conversational` | greetings, single words | 1 | paraphrase |

Returns a `RouterResult(intent, num_expansions, strategy)`.

---

### `backend/pipeline/expander.py`

**Class:** `QueryExpander`

Generates query variants using template-based strategies:

| Strategy | What it does |
|---|---|
| `paraphrase` | Synonym substitution (`best` → `top`, `recommended`) |
| `technical` | Appends qualifiers: `tutorial`, `documentation`, `guide` |
| `comparison` | Generates: `X alternatives`, `best X 2026`, `X vs alternatives` |
| `community` | Appends: `reddit`, `forum discussion`, `community experience` |
| `documentation` | Appends: `official docs`, `getting started`, `how to` |

The original query is always the first item. Output is capped at `pipeline.max_expanded_queries`.

---

### `backend/pipeline/retriever.py`

**Class:** `Retriever`

- Uses `asyncio.gather()` to fire all expanded queries at SearXNG **in parallel** — so N queries take ≈ the time of 1 query.
- Timeout per query: `pipeline.search_timeout_seconds` (default 2s).
- Retry: 1 automatic retry on timeout/HTTP error before giving up.
- Degradation: a failed query contributes an empty list — the pipeline continues with partial results.
- Shared `httpx.AsyncClient` created at startup (not per-request) for connection pooling.

---

### `backend/pipeline/aggregator.py` + `backend/utils/url_utils.py`

**Deduplication strategy:**

1. **Normalize** every URL: strip UTM/tracking params, lowercase domain, remove `www.`, strip trailing slash.
2. **Deduplicate** by normalized URL — first occurrence wins for title/URL.
3. **Merge snippets** from duplicate URLs: keep the longest as base, append unique sentences from others.
4. **Collect provenance**: which expanded queries and source engines contributed each result.

UTM params stripped: `utm_source`, `utm_medium`, `utm_campaign`, `utm_term`, `utm_content`, `ref`, `source`, `fbclid`, `gclid`, and more.

---

### `backend/pipeline/embedder.py` + `backend/models/embedder_model.py`

**Near-duplicate clustering (Phase 5):**

Purpose: URL dedup misses identical content at different URLs. Embedding clustering catches these.

Algorithm:
1. For each result, look up embedding in `EmbeddingCache` by `sha256(url)`. On miss, compute `embed(title + " " + snippet)`.
2. Compute cosine similarity matrix (fast dot product since embeddings are L2-normalized).
3. Union-Find clustering: results with similarity > `dedup_similarity_threshold` (default 0.92) are grouped.
4. From each cluster, keep the result with the longest snippet as the representative.

**Model:** `BAAI/bge-small-en` — fast (30ms/batch), 384-dim embeddings, English.

**Cache:** `EmbeddingCache` is an in-memory dict keyed by `sha256(url)`. Embeddings are deterministic so no TTL is needed. Persists for the lifetime of the process (across requests).

---

### `backend/pipeline/reranker.py` + `backend/models/reranker_model.py`

**Cross-encoder reranking (Phase 6 — the core ML step):**

- Input: original query + all results from the clustering stage.
- For each result: construct `"title. snippet"` as the document string.
- Pass all `(query, document)` pairs to `CrossEncoder.predict()` in one batch.
- Sort descending by score, return top `reranker.top_k`.
- CPU-bound inference is wrapped in `asyncio.to_thread()` to avoid blocking the event loop.

**Model:** `BAAI/bge-reranker-base` — cross-encoder trained on MS MARCO. Scores are logits (not bounded to [0,1]); the frontend applies sigmoid for display.

---

### `backend/pipeline/synthesizer.py` + `backend/models/llm_model.py`

**LLM synthesis (Phase 7):**

Prompt template (simplified):
```
[SYSTEM: cite sources inline as [N], don't go beyond the sources]

Query: {query}

Sources:
[1] {title_1}
{snippet_1}

[2] ...

Answer:
```

- Takes top `llm.max_sources_in_prompt` results (default 8).
- Parses `[N]` markers in the response and maps them to citation URLs.
- Returns `None` if synthesis fails, times out, or produces no citations (when `require_citations: true`).
- Pipeline continues gracefully without an answer.

**Model:** `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` — 1.5B param model, ~940 MB on disk. Runs on CPU (~500ms) or GPU (~150ms with `n_gpu_layers: 35`).

---

### `backend/logging/`

**`query_logger.py`:** Appends one JSONL record per search to `data/logs/queries-{YYYY-MM-DD}.jsonl`. Records include pipeline metadata: expanded queries, stage latencies, reranker scores, and whether an answer was generated.

**`feedback.py`:** Appends thumbs-up/down and click events to separate JSONL files. These are the ground-truth signals for future reranker fine-tuning.

---

## Startup Sequence (`backend/main.py` lifespan)

```
1. Load Settings from .env
2. Load ModelConfig from model_config.yaml
3. Create shared httpx.AsyncClient
4. Load EmbedderModel  (sentence-transformers, ~1-3s)
5. Load RerankerModel  (sentence-transformers, ~1-3s)
6. Load LLMModel       (llama-cpp-python, ~5-15s depending on model size)
7. Instantiate pipeline components (Router, Expander, Retriever, …)
8. Instantiate QueryLogger + FeedbackLogger
9. Mount frontend static files
```

All models are loaded **once** and stored on `app.state`. They are reused across all requests with zero cold-start overhead.

---

## Latency Budget

| Stage | Target | Bottleneck |
|---|---|---|
| Router | <10ms | Regex, zero model |
| Expansion | <20ms | Template strings |
| Search | 300–1200ms | Network (SearXNG); parallelized |
| Aggregation | <20ms | Pure Python |
| Embedding | <100ms | sentence-transformers + cache |
| Reranking | 50–150ms | Cross-encoder batch inference |
| Synthesis | 300–800ms | GGUF LLM on CPU |
| **Total (no LLM)** | **~800ms** | |
| **Total (with LLM)** | **~1.5–2.5s** | |

---

## Key Design Decisions

**Why no build step for the frontend?**
The frontend is plain HTML/CSS/JS served as static files by FastAPI. No Node.js, no bundler. This makes deployment trivial (`uvicorn backend.main:app`) and keeps the dev loop fast.

**Why Union-Find for clustering?**
It's O(n²) for the similarity matrix, which is fine for the 20–60 results we typically have after URL dedup. It's simpler and faster than HDBSCAN for this scale. If you're regularly seeing 200+ results, swap in a proper clustering algorithm.

**Why llama-cpp-python instead of Ollama?**
Direct GGUF loading avoids running a separate server process, simplifies deployment, and keeps all model loading inside the FastAPI lifespan. Tradeoff: harder to hot-swap models (requires a restart).

**Why `asyncio.to_thread()` for ML inference?**
The embedder and reranker use PyTorch under the hood — synchronous CPU-bound operations. Wrapping them in `asyncio.to_thread()` yields the event loop back while they run, so the server stays responsive to other requests during inference.

**Why JSONL for logs?**
Append-only JSONL is crash-safe (each record is a complete line), trivially parseable with `jq` or pandas, and rotation is just opening a new file. No database dependency.

---

## Extending YASS-MAN

### Add a new expansion strategy
1. Write a `_my_strategy_expand(query: str) -> list[str]` function in `expander.py`.
2. Add it to `_ALL_GENERATORS` and `_STRATEGY_GENERATORS`.

### Swap the reranker
Edit `model_config.yaml`:
```yaml
reranker:
  model: bge-reranker-large
  repo: BAAI/bge-reranker-large
```
Run `python scripts/download_models.py` and restart.

### Add domain filtering
Add a `BLOCKED_DOMAINS` list to `Settings` and filter in `aggregator.py` before dedup.

### Fine-tune the reranker on your feedback data
Parse `data/logs/feedback.jsonl` + `clicks.jsonl` to build `(query, url, label)` triplets, then fine-tune `bge-reranker-base` with a cross-encoder training loop. Update `model_config.yaml` to point to your fine-tuned model.
