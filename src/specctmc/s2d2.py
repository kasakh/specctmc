"""Finite event laws and exact maximal coupling for an S2D2 proposal.

The proposal uses one block diffusion forward at the current state. Token rows
are normalized independently. A masked position is sampled from an explicit
normalized confidence law. Later draft events reuse the same frozen rows.

The exact target is a natural event law. It samples a currently masked position
uniformly, then samples its replacement token from the normalized current state
logit row. The mask token is excluded because an event must leave the mask state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch

Event = Tuple[int, int]


def _as_bool_vector(active_mask: torch.Tensor) -> torch.Tensor:
    active = torch.as_tensor(active_mask, dtype=torch.bool)
    if active.ndim != 1:
        raise ValueError("active_mask must be one dimensional")
    if not bool(active.any()):
        raise ValueError("at least one position must be active")
    return active


def _normalize_nonnegative(values: torch.Tensor, name: str) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} contains a nonfinite value")
    if bool((values < 0).any()):
        raise ValueError(f"{name} contains a negative value")
    total = values.sum()
    if not bool(total > 0):
        raise ValueError(f"{name} has zero total mass")
    return values / total


def normalized_token_rows(
    logits: torch.Tensor,
    temperature: float = 1.0,
    forbidden_token_ids: Iterable[int] = (),
) -> torch.Tensor:
    """Return normalized token rows.

    Temperature zero denotes a greedy sampling law. Its row is a point mass with
    selected probability one. This is intentionally not the selected token's
    softmax confidence.
    """
    values = torch.as_tensor(logits, dtype=torch.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape positions by vocabulary")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("logits contain a nonfinite value")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")

    filtered = values.clone()
    forbidden = sorted(set(int(token) for token in forbidden_token_ids))
    for token in forbidden:
        if token < 0 or token >= filtered.shape[1]:
            raise ValueError("forbidden token identifier is outside the vocabulary")
        filtered[:, token] = -torch.inf
    if bool(torch.isneginf(filtered).all(dim=1).any()):
        raise ValueError("a token row has no allowed support")

    if temperature == 0.0:
        chosen = filtered.argmax(dim=1)
        rows = torch.zeros_like(filtered)
        rows.scatter_(1, chosen[:, None], 1.0)
        return rows

    rows = torch.softmax(filtered / float(temperature), dim=1)
    row_sums = rows.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(rows).all()) or not bool((row_sums > 0).all()):
        raise ValueError("softmax did not produce finite positive row mass")
    return rows / row_sums


def normalized_position_law(
    active_mask: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize position scores over active positions only."""
    active = _as_bool_vector(active_mask)
    if scores is None:
        weights = active.to(dtype=torch.float64)
    else:
        weights = torch.as_tensor(scores, dtype=torch.float64)
        if weights.shape != active.shape:
            raise ValueError("position scores and active_mask must have equal shape")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("position scores must be finite and nonnegative")
        weights = torch.where(active, weights, torch.zeros_like(weights))
    return _normalize_nonnegative(weights, "active position scores")


@dataclass(frozen=True)
class EventLaw:
    """A normalized distribution over position and token events."""

    position_probs: torch.Tensor
    token_probs: torch.Tensor
    joint_probs: torch.Tensor
    active_mask: torch.Tensor

    def __post_init__(self) -> None:
        position = torch.as_tensor(self.position_probs, dtype=torch.float64)
        token = torch.as_tensor(self.token_probs, dtype=torch.float64)
        joint = torch.as_tensor(self.joint_probs, dtype=torch.float64)
        active = torch.as_tensor(self.active_mask, dtype=torch.bool).clone()
        if position.ndim != 1 or token.ndim != 2:
            raise ValueError("position_probs and token_probs have invalid rank")
        if token.shape[0] != position.shape[0] or active.shape != position.shape:
            raise ValueError("event law shapes are inconsistent")
        if joint.shape != token.shape:
            raise ValueError("joint_probs must match token_probs shape")
        expected = position[:, None] * token
        if not torch.allclose(joint, expected, atol=1e-12, rtol=1e-10):
            raise ValueError("joint_probs do not factor into position and token laws")
        if not torch.allclose(position.sum(), position.new_tensor(1.0), atol=1e-12):
            raise ValueError("position law is not normalized")
        row_ones = torch.ones(token.shape[0], dtype=torch.float64, device=token.device)
        if not torch.allclose(token.sum(dim=1), row_ones, atol=1e-12):
            raise ValueError("token rows are not normalized")
        if not torch.allclose(joint.sum(), joint.new_tensor(1.0), atol=1e-12):
            raise ValueError("joint event law is not normalized")
        if bool((position[~active] != 0).any()):
            raise ValueError("inactive positions carry probability")
        object.__setattr__(self, "position_probs", position)
        object.__setattr__(self, "token_probs", token)
        object.__setattr__(self, "joint_probs", joint)
        object.__setattr__(self, "active_mask", active)

    @property
    def event_count(self) -> int:
        return int(self.joint_probs.numel())

    def mass(self, event: Event) -> float:
        position, token = event
        return float(self.joint_probs[position, token].item())


def event_law(position_probs: torch.Tensor, token_probs: torch.Tensor, active_mask: torch.Tensor) -> EventLaw:
    position = torch.as_tensor(position_probs, dtype=torch.float64)
    token = torch.as_tensor(token_probs, dtype=torch.float64)
    return EventLaw(position, token, position[:, None] * token, torch.as_tensor(active_mask, dtype=torch.bool))


def natural_event_target(
    logits: torch.Tensor,
    active_mask: torch.Tensor,
    temperature: float = 1.0,
    forbidden_token_ids: Iterable[int] = (),
) -> EventLaw:
    """Build the exact natural event target from current state logits."""
    token = normalized_token_rows(logits, temperature, forbidden_token_ids)
    position = normalized_position_law(active_mask)
    return event_law(position, token, active_mask)


def s2d2_proposal_law(
    frozen_logits: torch.Tensor,
    active_mask: torch.Tensor,
    token_temperature: float = 1.0,
    forbidden_token_ids: Iterable[int] = (),
    position_mode: str = "confidence",
) -> EventLaw:
    """Build a normalized S2D2 inspired law from one frozen block forward."""
    token = normalized_token_rows(frozen_logits, token_temperature, forbidden_token_ids)
    if position_mode == "confidence":
        scores = token.max(dim=1).values
    elif position_mode == "uniform":
        scores = None
    else:
        raise ValueError("position_mode must be confidence or uniform")
    position = normalized_position_law(active_mask, scores)
    return event_law(position, token, active_mask)


def _sample_categorical(probs: torch.Tensor, uniform: float) -> int:
    if not 0.0 <= uniform < 1.0:
        raise ValueError("uniform draw must lie in [0, 1)")
    flat = _normalize_nonnegative(torch.as_tensor(probs, dtype=torch.float64).reshape(-1), "categorical law")
    cdf = flat.cumsum(dim=0)
    index = int(torch.searchsorted(cdf, cdf.new_tensor(uniform), right=False).item())
    return min(index, flat.numel() - 1)


def sample_event(law: EventLaw, uniform: float) -> Event:
    flat_index = _sample_categorical(law.joint_probs, uniform)
    vocabulary = law.joint_probs.shape[1]
    return flat_index // vocabulary, flat_index % vocabulary


@dataclass(frozen=True)
class DraftEvent:
    event: Event
    selected_probability: float
    law: EventLaw


@dataclass(frozen=True)
class CouplingResult:
    event: Event
    proposed_event: Event
    accepted: bool
    acceptance_probability: float
    proposal_probability: float
    target_probability: float
    residual_mass: float
    proposal_law: EventLaw
    target_law: EventLaw


def draft_from_one_forward(
    logits: torch.Tensor,
    active_mask: torch.Tensor,
    depth: int,
    uniforms: Sequence[float],
    token_temperature: float = 1.0,
    forbidden_token_ids: Iterable[int] = (),
    position_mode: str = "confidence",
) -> Tuple[DraftEvent, ...]:
    """Draft up to depth events while reusing one frozen logit tensor."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    active = _as_bool_vector(active_mask).clone()
    count = min(depth, int(active.sum().item()))
    if len(uniforms) < count:
        raise ValueError("one proposal uniform is required per drafted event")
    token = normalized_token_rows(logits, token_temperature, forbidden_token_ids)
    if position_mode == "confidence":
        scores = token.max(dim=1).values
    elif position_mode == "uniform":
        scores = None
    else:
        raise ValueError("position_mode must be confidence or uniform")
    records = []
    for step in range(count):
        position = normalized_position_law(active, scores)
        law = event_law(position, token, active)
        chosen = sample_event(law, float(uniforms[step]))
        records.append(DraftEvent(chosen, law.mass(chosen), law))
        active[chosen[0]] = False
    return tuple(records)


def maximal_coupling_step(
    proposed: DraftEvent,
    target: EventLaw,
    acceptance_uniform: float,
    residual_uniform: float,
    tolerance: float = 1e-12,
) -> CouplingResult:
    """Accept a proposal or sample the positive residual of target minus proposal."""
    q = proposed.law
    if q.joint_probs.shape != target.joint_probs.shape:
        raise ValueError("proposal and target event spaces differ")
    if not torch.equal(q.active_mask, target.active_mask):
        raise ValueError("proposal and target active supports differ")
    q_mass = q.mass(proposed.event)
    p_mass = target.mass(proposed.event)
    if q_mass <= 0:
        raise ValueError("the sampled proposal event has zero proposal mass")
    acceptance = min(1.0, p_mass / q_mass)
    residual = (target.joint_probs - q.joint_probs).clamp_min(0.0)
    residual_mass = float(residual.sum().item())
    accepted = float(acceptance_uniform) < acceptance
    if accepted:
        output = proposed.event
    else:
        if residual_mass <= tolerance:
            raise RuntimeError("a rejection occurred with no positive residual mass")
        flat = _sample_categorical(residual / residual_mass, float(residual_uniform))
        vocabulary = residual.shape[1]
        output = flat // vocabulary, flat % vocabulary
    return CouplingResult(
        event=output,
        proposed_event=proposed.event,
        accepted=accepted,
        acceptance_probability=acceptance,
        proposal_probability=q_mass,
        target_probability=p_mass,
        residual_mass=residual_mass,
        proposal_law=q,
        target_law=target,
    )


def correct_draft(
    initial_active_mask: torch.Tensor,
    draft: Sequence[DraftEvent],
    target_law_for_prefix: Callable[[Tuple[Event, ...], torch.Tensor], EventLaw],
    acceptance_uniforms: Sequence[float],
    residual_uniforms: Sequence[float],
) -> Tuple[CouplingResult, ...]:
    """Correct a draft until its first rejection, returning an exact event prefix."""
    if len(acceptance_uniforms) < len(draft) or len(residual_uniforms) < len(draft):
        raise ValueError("one acceptance and residual uniform is required per draft event")
    active = _as_bool_vector(initial_active_mask).clone()
    accepted_prefix = []
    results = []
    for index, proposed in enumerate(draft):
        if not torch.equal(proposed.law.active_mask, active):
            raise ValueError("recorded proposal support does not match the corrected prefix")
        target = target_law_for_prefix(tuple(accepted_prefix), active.clone())
        result = maximal_coupling_step(
            proposed,
            target,
            float(acceptance_uniforms[index]),
            float(residual_uniforms[index]),
        )
        results.append(result)
        active[result.event[0]] = False
        if not result.accepted:
            break
        accepted_prefix.append(result.event)
    return tuple(results)


def full_probability_record(record: DraftEvent) -> dict:
    """Return every proposal probability, not only the selected token confidence."""
    return {
        "event": list(record.event),
        "selected_probability": record.selected_probability,
        "active_mask": record.law.active_mask.tolist(),
        "position_probs": record.law.position_probs.tolist(),
        "token_probs": record.law.token_probs.tolist(),
        "joint_probs": record.law.joint_probs.tolist(),
        "event_count": record.law.event_count,
    }
