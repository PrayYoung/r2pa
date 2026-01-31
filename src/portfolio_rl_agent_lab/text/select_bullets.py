from __future__ import annotations

from typing import List
import numpy as np


def select_representative_bullets(
    texts: List[str],
    emb: np.ndarray,
    k: int = 3,
    lambda_diversity: float = 0.7,
) -> List[str]:
    """
    Select k representative but diverse bullets using MMR (Maximal Marginal Relevance).

    emb: (N, D) normalized embeddings.
    lambda_diversity:
        - closer to 1.0 -> prioritize relevance to centroid
        - closer to 0.0 -> prioritize diversity
    """
    if not texts or emb.size == 0:
        return []

    # Compute centroid (overall topic)
    centroid = emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    # Similarity to centroid
    sim_to_centroid = emb @ centroid

    selected = []
    selected_idx = []

    # 1) Pick the most representative bullet first
    first = int(np.argmax(sim_to_centroid))
    selected.append(texts[first])
    selected_idx.append(first)

    # 2) Iteratively pick bullets with MMR
    for _ in range(1, min(k, len(texts))):
        scores = []

        for i in range(len(texts)):
            if i in selected_idx:
                scores.append(-np.inf)
                continue

            # Max similarity to already selected bullets
            sim_to_selected = max(
                emb[i] @ emb[j] for j in selected_idx
            )

            mmr_score = (
                lambda_diversity * sim_to_centroid[i]
                - (1.0 - lambda_diversity) * sim_to_selected
            )
            scores.append(mmr_score)

        next_i = int(np.argmax(scores))
        selected.append(texts[next_i])
        selected_idx.append(next_i)

    return selected
