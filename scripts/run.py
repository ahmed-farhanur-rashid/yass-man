#!/usr/bin/env python3
"""
YASS-MAN launcher with pre-flight checks.

Usage:
    python scripts/run.py           # production (reload=False)
    python scripts/run.py --dev     # development (reload=True, debug logging)
    python scripts/run.py --check   # only run checks, don't start server

Pre-flight checks:
  1. SearXNG connectivity
  2. Model files present (embedder dir, reranker dir, GGUF file)
  3. .env exists
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_env() -> bool:
    env_path = Path(".env")
    if not env_path.exists():
        print("  ✗ .env not found — copy .env.example to .env and set SEARXNG_URL")
        return False
    print("  ✓ .env found")
    return True


def check_model_files(cfg) -> bool:
    ok = True

    embedder_dir = Path("models") / cfg.embedder.model
    if embedder_dir.exists():
        print(f"  ✓ Embedder: {embedder_dir}")
    else:
        print(f"  ✗ Embedder not found: {embedder_dir}")
        print(f"    Run: python scripts/download_models.py")
        ok = False

    reranker_dir = Path("models") / cfg.reranker.model
    if reranker_dir.exists():
        print(f"  ✓ Reranker: {reranker_dir}")
    else:
        print(f"  ✗ Reranker not found: {reranker_dir}")
        print(f"    Run: python scripts/download_models.py")
        ok = False

    if cfg.llm.enabled:
        llm_path = Path(cfg.llm.path)
        if llm_path.exists():
            size_mb = llm_path.stat().st_size / (1024 ** 2)
            print(f"  ✓ LLM: {llm_path} ({size_mb:.0f} MB)")
        else:
            print(f"  ✗ LLM not found: {llm_path}")
            print(f"    Run: python scripts/download_models.py")
            print(f"    Or disable with: llm.enabled: false in model_config.yaml")
            ok = False
    else:
        print("  ─ LLM disabled in model_config.yaml")

    return ok


async def check_searxng(url: str) -> bool:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params={"q": "test", "format": "json"})
            if resp.status_code == 200:
                print(f"  ✓ SearXNG reachable: {url}")
                return True
            else:
                print(f"  ✗ SearXNG returned HTTP {resp.status_code}: {url}")
                return False
    except Exception as exc:
        print(f"  ✗ SearXNG not reachable ({exc}): {url}")
        print(f"    Set SEARXNG_URL in .env to a working instance")
        print(f"    Find instances: https://searx.space (look for ✓ API column)")
        return False


async def preflight() -> bool:
    print("=== YASS-MAN Pre-flight Checks ===\n")
    all_ok = True

    # .env
    print("[1] Environment:")
    all_ok &= check_env()

    # Load config (requires .env)
    try:
        from backend.config import get_settings
        from backend.model_config_loader import load_model_config
        settings = get_settings()
        cfg = load_model_config(str(settings.model_config_path))
    except Exception as exc:
        print(f"\n  ✗ Config error: {exc}")
        return False

    # Models
    print("\n[2] Model files:")
    all_ok &= check_model_files(cfg)

    # SearXNG
    print("\n[3] SearXNG connectivity:")
    all_ok &= await check_searxng(settings.searxng_search_url)

    print()
    if all_ok:
        print("✓ All checks passed — ready to start.\n")
    else:
        print("✗ Some checks failed — fix the issues above before starting.\n")
    return all_ok


async def main() -> None:
    parser = argparse.ArgumentParser(description="YASS-MAN launcher")
    parser.add_argument("--dev",   action="store_true", help="Enable hot reload and DEBUG logging")
    parser.add_argument("--check", action="store_true", help="Only run pre-flight checks, don't start")
    args = parser.parse_args()

    ok = await preflight()

    if args.check:
        sys.exit(0 if ok else 1)

    if not ok:
        ans = input("Checks failed. Start anyway? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit(1)

    try:
        import uvicorn
        from backend.config import get_settings
        settings = get_settings()
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    log_level = "debug" if args.dev else "info"
    reload    = args.dev

    print(f"Starting YASS-MAN on http://{settings.host}:{settings.port}")
    if args.dev:
        print("  Hot reload enabled (--dev mode)\n")

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    asyncio.run(main())
