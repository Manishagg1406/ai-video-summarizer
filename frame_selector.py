"""
Keyframe Selection: pick the top-K most important frames
using temporal model scores + diversity (MMR).
"""
import torch
import numpy as np
from typing import List, Tuple


def select_keyframes_by_score(scores: torch.Tensor,
                               timestamps: List[float],
                               top_k: int = 8) -> List[int]:
    """Simple top-K selection by importance score."""
    k = min(top_k, len(scores))
    indices = torch.topk(scores, k).indices.tolist()
    return sorted(indices)


def select_keyframes_mmr(scores: torch.Tensor,
                          embeddings: torch.Tensor,
                          top_k: int = 8,
                          lambda_: float = 0.6) -> List[int]:
    """
    Maximal Marginal Relevance: balance importance + diversity.
    lambda_ controls relevance vs. diversity trade-off.
    """
    scores_np = scores.numpy()
    emb_np    = embeddings.numpy()

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8
    emb_norm = emb_np / norms

    selected, remaining = [], list(range(len(scores_np)))

    for _ in range(min(top_k, len(scores_np))):
        if not remaining:
            break
        if not selected:
            # First pick: highest score
            best = max(remaining, key=lambda i: scores_np[i])
        else:
            sel_emb = emb_norm[selected]
            best, best_val = None, -np.inf
            for i in remaining:
                rel  = scores_np[i]
                sim  = float(np.max(emb_norm[i] @ sel_emb.T))
                val  = lambda_ * rel - (1 - lambda_) * sim
                if val > best_val:
                    best_val, best = val, i
        selected.append(best)
        remaining.remove(best)

    return sorted(selected)


def scores_to_segments(scores: torch.Tensor,
                        timestamps: List[float],
                        threshold: float = 0.5,
                        min_gap: float = 2.0) -> List[Tuple[float, float]]:
    """
    Convert frame scores to time segments (start, end) for highlight reel.
    Merges consecutive high-scoring frames separated by less than min_gap.
    """
    high = [(timestamps[i], scores[i].item())
            for i in range(len(timestamps)) if scores[i] >= threshold]
    if not high:
        return []

    segments, seg_start, prev_t = [], high[0][0], high[0][0]
    for t, _ in high[1:]:
        if t - prev_t > min_gap:
            segments.append((seg_start, prev_t))
            seg_start = t
        prev_t = t
    segments.append((seg_start, prev_t))
    return segments
