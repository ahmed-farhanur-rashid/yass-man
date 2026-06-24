#!/usr/bin/env python3
"""
YASS-MAN Latency Benchmark.

Runs 10 fixed test queries through the full pipeline and reports
min / avg / p95 latency for each stage, flagging any that exceed targets.

Usage:
    python scripts/benchmark.py [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

TEST_QUERIES = [
    "best GPU for AI 2026",
    "Python vs JavaScript performance",
    "how to fix Docker permission denied error",
    "what is transformer architecture",
    "FastAPI tutorial beginners",
    "difference between TCP and UDP",
    "Rust programming language overview",
    "machine learning model deployment guide",
    "PostgreSQL vs MongoDB comparison",
    "how to set up SSH keys on Linux",
]

# Latency targets in ms
TARGETS = {
    "router":      10,
    "expansion":   20,
    "search":    1200,
    "aggregation": 20,
    "embedding":  100,
    "rerank":     150,
    "synthesis":  800,
    "total":     2500,
}


async def run_query(client: httpx.AsyncClient, base_url: str, query: str) -> dict:
    resp = await client.get(f"{base_url}/search", params={"q": query}, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def percentile(data: list[float], p: float) -> float:
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1
    if hi >= len(data):
        return data[-1]
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


async def main(base_url: str) -> None:
    print(f"YASS-MAN Benchmark — {len(TEST_QUERIES)} queries → {base_url}\n")

    stage_data: dict[str, list[float]] = {s: [] for s in TARGETS}

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(TEST_QUERIES, 1):
            print(f"  [{i:02d}/{len(TEST_QUERIES)}] {q[:60]}")
            try:
                data = await run_query(client, base_url, q)
                lat = data.get("latency_ms", {})
                for stage in TARGETS:
                    if stage in lat:
                        stage_data[stage].append(lat[stage])
            except Exception as exc:
                print(f"         ERROR: {exc}")

    print("\n" + "─" * 72)
    print(f"{'Stage':<14} {'Min':>8} {'Avg':>8} {'P95':>8} {'Target':>8}  {'Status'}")
    print("─" * 72)

    any_fail = False
    for stage, target in TARGETS.items():
        vals = stage_data.get(stage, [])
        if not vals:
            print(f"{stage:<14} {'N/A':>8}")
            continue
        mn  = min(vals)
        avg = statistics.mean(vals)
        p95 = percentile(vals, 95)
        ok  = p95 <= target
        flag = "✓" if ok else "✗ SLOW"
        if not ok:
            any_fail = True
        print(f"{stage:<14} {mn:>7.0f}ms {avg:>7.0f}ms {p95:>7.0f}ms {target:>7}ms  {flag}")

    print("─" * 72)
    if any_fail:
        print("\n⚠ Some stages exceed latency targets. Consider:")
        print("  • Set n_gpu_layers > 0 in model_config.yaml (if CUDA available)")
        print("  • Set llm.enabled: false to skip synthesis")
        print("  • Use a faster SearXNG instance")
    else:
        print("\n✓ All stages within latency targets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    asyncio.run(main(args.url))
