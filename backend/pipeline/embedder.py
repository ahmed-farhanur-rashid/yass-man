"""
Embedding + Clustering Stage — Phase 5.

Removes near-duplicate results that URL dedup missed
(same content served at different URLs).

Algorithm:
1. Embed title + snippet for each result (with cache lookup).
2. Compute pairwise cosine similarities.
3. Union-Find clustering: group results with similarity > threshold.
4. Keep the result with the longest snippet from each cluster.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from backend.cache.embedding_cache import EmbeddingCache
from backend.model_config_loader import PipelineConfig
from backend.models.embedder_model import EmbedderModel
from backend.schemas.result import AggregatedResult

logger = logging.getLogger(__name__)


class EmbedderStage:
    """
    Near-duplicate clustering using embedding cosine similarity.
    """

    def __init__(
        self,
        model: EmbedderModel,
        cache: EmbeddingCache,
        config: PipelineConfig,
        enabled: bool = True,
    ) -> None:
        self._model = model
        self._cache = cache
        self._threshold = config.dedup_similarity_threshold
        self._enabled = enabled

    async def cluster(self, results: list[AggregatedResult]) -> list[AggregatedResult]:
        """
        Remove near-duplicates and return a smaller, cleaner list.
        If clustering is disabled, returns *results* unchanged.
        """
        if not self._enabled or len(results) <= 1:
            return results

        embeddings = await asyncio.to_thread(self._compute_embeddings, results)
        clusters = self._find_clusters(embeddings)
        deduplicated = self._pick_representatives(results, clusters)

        logger.info(
            "Embedder: %d results → %d after near-dedup (threshold=%.2f)",
            len(results),
            len(deduplicated),
            self._threshold,
        )
        return deduplicated

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_embeddings(self, results: list[AggregatedResult]) -> np.ndarray:
        """Return embedding matrix, using cache where available."""
        texts: list[str] = []
        cached_embeddings: dict[int, np.ndarray] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, r in enumerate(results):
            cached = self._cache.get(r.normalized_url)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(f"{r.title} {r.snippet}")

        # Batch-embed uncached texts
        if uncached_texts:
            new_embeddings = self._model.embed(uncached_texts)
            for local_i, global_i in enumerate(uncached_indices):
                emb = new_embeddings[local_i]
                cached_embeddings[global_i] = emb
                self._cache.set(results[global_i].normalized_url, emb)

        # Assemble in order
        dim = next(iter(cached_embeddings.values())).shape[0] if cached_embeddings else 1
        matrix = np.zeros((len(results), dim), dtype=np.float32)
        for i, emb in cached_embeddings.items():
            matrix[i] = emb

        return matrix

    def _find_clusters(self, embeddings: np.ndarray) -> list[int]:
        """
        Union-Find clustering.

        Returns a list where clusters[i] is the cluster representative index
        for result i. All results with the same representative belong to one cluster.
        """
        n = len(embeddings)
        # Cosine similarity via dot product (embeddings are L2-normalized)
        sim_matrix = embeddings @ embeddings.T

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self._threshold:
                    union(i, j)

        return [find(i) for i in range(n)]

    def _pick_representatives(
        self,
        results: list[AggregatedResult],
        clusters: list[int],
    ) -> list[AggregatedResult]:
        """From each cluster, keep the result with the longest snippet."""
        cluster_map: dict[int, list[int]] = {}
        for i, c in enumerate(clusters):
            cluster_map.setdefault(c, []).append(i)

        representatives: list[AggregatedResult] = []
        # Preserve original ordering by using the first-seen cluster head
        seen_clusters: set[int] = set()
        for i, c in enumerate(clusters):
            if c not in seen_clusters:
                seen_clusters.add(c)
                members = cluster_map[c]
                best = max(members, key=lambda idx: len(results[idx].snippet))
                representatives.append(results[best])

        return representatives
