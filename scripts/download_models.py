#!/usr/bin/env python3
"""
Download all models referenced in model_config.yaml.

Usage:
    python scripts/download_models.py

- Embedder + reranker: downloaded via huggingface_hub
- LLM (GGUF): downloaded with tqdm progress bar if path doesn't exist
- Skips any model already present
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def download_hf_model(repo: str, model_dir: Path, model_name: str) -> None:
    """Download a HuggingFace model via huggingface_hub snapshot_download."""
    target = model_dir / model_name
    if target.exists():
        logger.info("  ✓ Already exists: %s", target)
        return

    logger.info("  ↓ Downloading %s from %s …", model_name, repo)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        logger.info("  ✓ Saved to: %s", target)
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        raise
    except Exception as exc:
        logger.error("  ✗ Failed to download %s: %s", repo, exc)
        raise


def download_gguf(url: str, dest_path: Path) -> None:
    """Download a GGUF file with a tqdm progress bar."""
    if dest_path.exists():
        logger.info("  ✓ Already exists: %s", dest_path)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  ↓ Downloading %s …", dest_path.name)

    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        logger.error("requests and tqdm are required. Run: pip install requests tqdm")
        raise

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB

    with dest_path.open("wb") as fh, tqdm(
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest_path.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            size = fh.write(chunk)
            bar.update(size)

    logger.info("  ✓ Saved to: %s", dest_path)


def main() -> None:
    from backend.config import get_settings
    from backend.model_config_loader import load_model_config

    settings = get_settings()
    cfg = load_model_config(str(settings.model_config_path))
    model_dir = settings.model_dir

    logger.info("=== YASS-MAN Model Downloader ===")
    logger.info("Model directory: %s\n", model_dir.resolve())

    # ── Embedder ──────────────────────────────────────────────────────────────
    logger.info("[1/3] Embedder: %s", cfg.embedder.repo)
    download_hf_model(cfg.embedder.repo, model_dir, cfg.embedder.model)

    # ── Reranker ──────────────────────────────────────────────────────────────
    logger.info("\n[2/3] Reranker: %s", cfg.reranker.repo)
    download_hf_model(cfg.reranker.repo, model_dir, cfg.reranker.model)

    # ── LLM ───────────────────────────────────────────────────────────────────
    logger.info("\n[3/3] LLM: %s", cfg.llm.model)
    if not cfg.llm.enabled:
        logger.info("  ─ LLM is disabled in model_config.yaml, skipping.")
    elif cfg.llm.download_url:
        llm_path = Path(cfg.llm.path)
        download_gguf(cfg.llm.download_url, llm_path)
    else:
        logger.info("  ─ No download_url set for LLM, skipping.")

    logger.info("\n✓ All models ready.")


if __name__ == "__main__":
    main()
