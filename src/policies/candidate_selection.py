"""Select a low-residual proposal or request a full sampler fallback."""

import numpy as np


def select_candidate(scores, *, min_gap, max_residual=None):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (2,) or not 0 <= min_gap <= 1:
        raise ValueError("Expected two scores and a min_gap in [0, 1]")
    if max_residual is not None and (not np.isfinite(max_residual) or max_residual <= 0):
        raise ValueError("max_residual must be finite and positive")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        return {"selected_candidate_idx": None, "relative_gap": None, "fallback": True, "reason": "invalid_scores"}
    best = int(np.argmin(scores))
    gap = float((scores.max()-scores.min())/max(scores.max(), 1e-12))
    reason = "accepted"
    if max_residual is not None and scores[best] > max_residual:
        reason = "residual_limit"
    elif gap <= min_gap:
        reason = "ambiguous_scores"
    return {"selected_candidate_idx": best, "relative_gap": gap, "fallback": reason != "accepted", "reason": reason}
