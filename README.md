# YASS-MAN

**Yet Another SearXNG Search Meta AI Network**

A self-hosted, privacy-respecting, AI-enhanced search engine built on top of [SearXNG](https://github.com/searxng/searxng). YASS-MAN fans your query out into multiple reformulations, searches in parallel, deduplicates and reranks the results with a local cross-encoder, then synthesizes a cited answer with a local GGUF LLM — all running on your machine, no API keys, no telemetry.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/you/yass-man.git
cd yass-man
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on llama-cpp-python:** If `pip install` fails (especially on Windows), grab a pre-built wheel from [llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases) matching your Python version and CUDA/CPU build.

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set `SEARXNG_URL` to a live SearXNG instance:
- Find API-enabled public instances at [searx.space](https://searx.space) — look for the **✓ API** column.
- Or run your own: `docker run -p 8080:8080 searxng/searxng`

### 3. Download models

```bash
python scripts/download_models.py
```

This downloads (~500 MB total):
- `BAAI/bge-small-en` — embedder (~130 MB)
- `BAAI/bge-reranker-base` — cross-encoder (~280 MB)
- `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` — LLM (~940 MB)

### 4. Run

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000** — type a query, see AI-enhanced results.

---

## Configuration

All model settings are in `model_config.yaml`. All runtime settings are in `.env`.

### Disable the LLM (faster, no synthesis)

```yaml
# model_config.yaml
llm:
  enabled: false
```

### Use GPU acceleration

```yaml
llm:
  n_gpu_layers: 35   # or higher; 35 offloads most of the model on a 6GB VRAM GPU
```

### Swap to a better reranker

```yaml
reranker:
  model: bge-reranker-large
  repo: BAAI/bge-reranker-large
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/search?q=your+query` | GET | Full pipeline search |
| `/feedback` | POST | Submit relevance signal (up/down) |
| `/click` | POST | Log a result click |
| `/health` | GET | System health check |

### Search response

```json
{
  "query_id": "uuid",
  "query": "best GPU for AI",
  "expanded_queries": ["best GPU for machine learning", "..."],
  "answer": "As of 2026, the RTX 4070 Ti Super is widely recommended... [1]",
  "citations": [{ "index": 1, "title": "...", "url": "..." }],
  "results": [{ "title": "...", "url": "...", "snippet": "...", "score": 0.94 }],
  "latency_ms": { "router": 4, "expansion": 12, "search": 810, "total": 1509 }
}
```

---

## Running Tests

```bash
pytest tests/unit/ -v          # fast, no ML models needed
pytest tests/integration/ -v   # also no ML models (mocked)
```

---

## Benchmarking & Evaluation

```bash
# Latency benchmark (requires running server)
python scripts/benchmark.py

# Quality evaluation vs raw SearXNG
python scripts/eval.py
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full description of the pipeline, data flow, and design decisions.
