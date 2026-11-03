"""Static relaxed maximal-coupling verification for joint CTMC events."""

from __future__ import annotations

import math


def validate_relaxation_value(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("verification_mixture_lambda must be finite and lie in [0, 1)")
    return value


def relaxed_acceptance_probability(
    target_probability: float,
    draft_probability: float,
    verification_mixture_lambda: float,
) -> float:
    """Return maximal-coupling acceptance against the statewise P/D mixture.

    The values are conditional token probabilities at the proposed position. The
    uniform position factor is common to target and draft and cancels in the ratio.
    """
    target = float(target_probability)
    draft = float(draft_probability)
    relaxation = validate_relaxation_value(verification_mixture_lambda)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("target_probability must be finite and nonnegative")
    if not math.isfinite(draft) or draft <= 0.0:
        raise ValueError("draft_probability must be finite and strictly positive")
    if relaxation == 0.0:
        return min(1.0, target / draft)
    return min(1.0, relaxation + (1.0 - relaxation) * target / draft)
