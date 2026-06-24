#!/usr/bin/env python3
"""
YASS-MAN Offline Evaluation.

Compares Precision@5 and NDCG@10 between:
  (a) Raw SearXNG results
  (b) Full YASS-MAN pipeline

Uses 20 handcrafted queries with known relevant URLs.

Usage:
    python scripts/eval.py [--url http://localhost:8000] [--searxng https://searx.be]
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

# Ground truth: query → set of known-relevant URL substrings
GROUND_TRUTH: list[dict] = [
    {"query": "Python official documentation", "relevant": ["docs.python.org"]},
    {"query": "FastAPI documentation", "relevant": ["fastapi.tiangolo.com"]},
    {"query": "React getting started tutorial", "relevant": ["react.dev", "reactjs.org"]},
    {"query": "Docker installation guide", "relevant": ["docs.docker.com"]},
    {"query": "PostgreSQL official docs", "relevant": ["postgresql.org"]},
    {"query": "Rust programming language book", "relevant": ["doc.rust-lang.org"]},
    {"query": "NumPy documentation", "relevant": ["numpy.org"]},
    {"query": "Git basics tutorial", "relevant": ["git-scm.com", "atlassian.com/git"]},
    {"query": "SQLAlchemy ORM tutorial", "relevant": ["docs.sqlalchemy.org"]},
    {"query": "Kubernetes getting started", "relevant": ["kubernetes.io"]},
    {"query": "TypeScript handbook", "relevant": ["typescriptlang.org"]},
    {"query": "Linux command line tutorial", "relevant": ["linuxcommand.org", "tldp.org", "man7.org"]},
    {"query": "VS Code keyboard shortcuts", "relevant": ["code.visualstudio.com"]},
    {"query": "GitHub Actions documentation", "relevant": ["docs.github.com"]},
    {"query": "Redis data structures", "relevant": ["redis.io"]},
    {"query": "AWS S3 documentation", "relevant": ["docs.aws.amazon.com"]},
    {"query": "Nginx configuration guide", "relevant": ["nginx.org", "nginx.com/resources"]},
    {"query": "TensorFlow tutorials", "relevant": ["tensorflow.org"]},
    {"query": "Pandas DataFrame tutorial", "relevant": ["pandas.pydata.org"]},
    {"query": "JWT authentication explained", "relevant": ["jwt.io", "auth0.com/learn"]},
]


def is_relevant(url: str, relevant_patterns: list[str]) -> bool:
    url_lower = url.lower()
    return any(pat in url_lower for pat in relevant_patterns)


def precision_at_k(urls: list[str], relevant_patterns: list[str], k: int) -> float:
    top_k = urls[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for u in top_k if is_relevant(u, relevant_patterns))
    return hits / k


def ndcg_at_k(urls: list[str], relevant_patterns: list[str], k: int) -> float:
    top_k = urls[:k]
    if not top_k:
        return 0.0
    dcg = sum(
        (1 if is_relevant(u, relevant_patterns) else 0) / math.log2(i + 2)
        for i, u in enumerate(top_k)
    )
    # Ideal DCG: all relevant at top
    n_relevant = sum(1 for u in top_k if is_relevant(u, relevant_patterns))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / idcg if idcg > 0 else 0.0


async def fetch_yass(client: httpx.AsyncClient, base_url: str, query: str) -> list[str]:
    resp = await client.get(f"{base_url}/search", params={"q": query}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    return [r["url"] for r in data.get("results", [])]


async def fetch_raw_searxng(client: httpx.AsyncClient, searxng_url: str, query: str) -> list[str]:
    resp = await client.get(
        f"{searxng_url}/search",
        params={"q": query, "format": "json"},
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return [r.get("url", "") for r in data.get("results", [])]


async def main(base_url: str, searxng_url: str) -> None:
    print(f"YASS-MAN Evaluation — {len(GROUND_TRUTH)} queries\n")

    raw_p5: list[float] = []
    raw_ndcg: list[float] = []
    yass_p5: list[float] = []
    yass_ndcg: list[float] = []
    yass_wins = 0

    async with httpx.AsyncClient() as client:
        for item in GROUND_TRUTH:
            query = item["query"]
            relevant = item["relevant"]
            print(f"  Query: {query[:55]}")

            try:
                raw_urls  = await fetch_raw_searxng(client, searxng_url, query)
                yass_urls = await fetch_yass(client, base_url, query)
            except Exception as exc:
                print(f"    ERROR: {exc}")
                continue

            rp5  = precision_at_k(raw_urls,  relevant, 5)
            rn10 = ndcg_at_k(raw_urls,  relevant, 10)
            yp5  = precision_at_k(yass_urls, relevant, 5)
            yn10 = ndcg_at_k(yass_urls, relevant, 10)

            raw_p5.append(rp5);   raw_ndcg.append(rn10)
            yass_p5.append(yp5);  yass_ndcg.append(yn10)

            win = "✓" if yp5 >= rp5 else "✗"
            if yp5 >= rp5:
                yass_wins += 1
            print(f"    Raw   P@5={rp5:.2f}  NDCG@10={rn10:.2f}")
            print(f"    YASS  P@5={yp5:.2f}  NDCG@10={yn10:.2f}  {win}")

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    print("\n" + "═" * 56)
    print(f"{'Metric':<20} {'Raw SearXNG':>14} {'YASS-MAN':>14}")
    print("─" * 56)
    print(f"{'Avg Precision@5':<20} {avg(raw_p5):>14.4f} {avg(yass_p5):>14.4f}")
    print(f"{'Avg NDCG@10':<20} {avg(raw_ndcg):>14.4f} {avg(yass_ndcg):>14.4f}")
    print(f"{'YASS wins (P@5)':<20} {'-':>14} {yass_wins:>13}/{len(GROUND_TRUTH)}")
    print("═" * 56)

    target_wins = int(len(GROUND_TRUTH) * 0.75)
    if yass_wins >= target_wins:
        print(f"\n✓ MVP target met: YASS-MAN wins on {yass_wins}/{len(GROUND_TRUTH)} queries (target ≥{target_wins}).")
    else:
        print(f"\n✗ MVP target NOT met: {yass_wins}/{len(GROUND_TRUTH)} wins (need ≥{target_wins}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",      default="http://localhost:8000")
    parser.add_argument("--searxng",  default="https://searx.be")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.searxng))
