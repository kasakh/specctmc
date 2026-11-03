"""Approximate fixed-step solvers for absorbing-mask continuous-time Markov chains.

The rate function must match the staged DirectCTMC signature and must return rates
only for masked positions allowed by the active mask. These solvers approximate a
different output law from DirectCTMC. They do not provide an exactness guarantee.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from typing import Any, Optional

import torch

RateFunction = Callable[..., torch.Tensor]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_hash(x: torch.Tensor) -> str:
    payload = x.detach().to("cpu", non_blocking=False).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _model_device(model: Any, prompt: Optional[torch.Tensor]) -> torch.device:
    if prompt is not None:
        return prompt.device
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _initial_state(
    model: Any,
    prompt: Optional[torch.Tensor],
    batch_size: int,
    gen_length: int,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if gen_length <= 0:
        raise ValueError("gen_length must be positive")
    device = _model_device(model, prompt)
    prompt_length = 0
    if prompt is not None:
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("prompt must have shape [1, prompt_length]")
        prompt_length = int(prompt.shape[1])
    x = torch.full(
        (batch_size, prompt_length + gen_length),
        int(mask_id),
        dtype=torch.long,
        device=device,
    )
    if prompt is not None:
        x[:, :prompt_length] = prompt.to(device).expand(batch_size, -1)
    prompt_mask = torch.zeros_like(x, dtype=torch.bool)
    prompt_mask[:, :prompt_length] = True
    return x, prompt_mask, prompt_length


def _block_active_mask(
    x: torch.Tensor,
    prompt_length: int,
    block_index: int,
    block_length: int,
) -> torch.Tensor:
    active = torch.zeros_like(x, dtype=torch.bool)
    start = prompt_length + block_index * block_length
    active[:, start : start + block_length] = True
    return active


def _time_grid(steps: int, eps: float) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between zero and one")
    grid = [1.0 - (1.0 - eps) * index / steps for index in range(steps + 1)]
    grid[-1] = float(eps)
    if any(left <= right for left, right in zip(grid, grid[1:])):
        raise RuntimeError("the solver time grid must be strictly descending")
    return grid


def _new_metrics() -> dict[str, Any]:
    return {
        "sequential_target_rounds": 0,
        "total_target_rate_rows": 0,
        "total_model_input_rows": 0,
        "target_padding_rows": 0,
        "target_forward_seconds": 0.0,
        "configured_steps_per_block": 0,
        "completed_fixed_steps": 0,
        "accepted_positions": 0,
        "conflicting_positions_discarded": 0,
        "sampled_reaction_count": 0,
        "predictor_accepted_positions": 0,
        "predictor_conflicting_positions_discarded": 0,
        "predictor_sampled_reaction_count": 0,
        "trapezoidal_clipped_entries": 0,
        "trapezoidal_clipped_rate_mass": 0.0,
        "terminal_cleanup_positions": 0,
        "terminal_cleanup_rows": 0,
    }


def _rate_coordinates(
    rates: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
) -> tuple[int, torch.Tensor]:
    if rates.ndim != 3 or rates.shape[0] != x.shape[0]:
        raise RuntimeError("rate function must return [batch, positions, vocabulary]")
    if rates.shape[1] == x.shape[1]:
        return 0, x
    suffix_length = x.shape[1] - prompt_length
    if rates.shape[1] == suffix_length:
        return prompt_length, x[:, prompt_length:]
    raise RuntimeError("rate positions match neither the full sequence nor generated suffix")


def _validate_rates(
    rates: torch.Tensor,
    x: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
    prompt_length: int,
    mask_id: int,
    vocab_size: int,
) -> int:
    offset, state_view = _rate_coordinates(rates, x, prompt_length)
    if rates.shape[-1] != vocab_size:
        raise RuntimeError("rate vocabulary differs from vocab_size")
    if not 0 <= mask_id < vocab_size:
        raise ValueError("mask_id is outside the vocabulary")
    if not bool(torch.isfinite(rates).all().item()) or bool((rates < 0).any().item()):
        raise RuntimeError("rates must be finite and nonnegative")
    active_view = active_mask[:, offset : offset + rates.shape[1]]
    prompt_view = prompt_mask[:, offset : offset + rates.shape[1]]
    mutable = active_view & (~prompt_view) & (state_view == mask_id)
    row_sums = rates.sum(dim=-1)
    if bool(torch.count_nonzero(row_sums[~mutable]).item()):
        raise RuntimeError("rate function assigned mass outside mutable masked positions")
    if mutable.any() and bool((row_sums[mutable] <= 0).any().item()):
        raise RuntimeError("a mutable masked position has zero total rate")
    if bool(torch.count_nonzero(rates[..., mask_id]).item()):
        raise RuntimeError("rate function assigned mass to the mask token")
    diagonal = rates.gather(-1, state_view.unsqueeze(-1)).squeeze(-1)
    if bool(torch.count_nonzero(diagonal).item()):
        raise RuntimeError("rate function assigned mass to the current token")
    return offset


def _evaluate_rates(
    rate_function: RateFunction,
    model: Any,
    x: torch.Tensor,
    times: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
    prompt_length: int,
    metrics: dict[str, Any],
    mask_id: int,
    vocab_size: int,
    temperature: float,
    cfg_scale: float,
    target_evaluation_batch_size: int,
) -> tuple[torch.Tensor, int]:
    rows = x.shape[0]
    physical_rows = int(target_evaluation_batch_size)
    if physical_rows <= 0:
        raise ValueError("target_evaluation_batch_size must be positive")
    chunks: list[torch.Tensor] = []
    offsets: list[int] = []
    for start in range(0, rows, physical_rows):
        stop = min(start + physical_rows, rows)
        real_rows = stop - start
        indices = torch.arange(start, stop, device=x.device)
        padding_rows = physical_rows - real_rows
        if padding_rows:
            duplicate = torch.full(
                (padding_rows,), stop - 1, device=x.device, dtype=torch.long
            )
            indices = torch.cat((indices, duplicate), dim=0)
        eval_x = x.index_select(0, indices)
        eval_times = times.index_select(0, indices)
        eval_prompt_mask = prompt_mask.index_select(0, indices)
        eval_active_mask = active_mask.index_select(0, indices)
        _sync(x.device)
        started = time.perf_counter()
        evaluated = rate_function(
            model=model,
            x=eval_x,
            t=eval_times,
            prompt_mask=eval_prompt_mask,
            active_mask=eval_active_mask,
            mask_id=mask_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
        )
        _sync(x.device)
        elapsed = time.perf_counter() - started
        evaluated = evaluated[:real_rows]
        offset = _validate_rates(
            evaluated,
            x[start:stop],
            prompt_mask[start:stop],
            active_mask[start:stop],
            prompt_length,
            mask_id,
            vocab_size,
        )
        chunks.append(evaluated)
        offsets.append(offset)
        metrics["sequential_target_rounds"] += 1
        metrics["total_target_rate_rows"] += real_rows
        metrics["target_padding_rows"] += padding_rows
        metrics["total_model_input_rows"] += physical_rows * (2 if cfg_scale > 0 else 1)
        metrics["target_forward_seconds"] += elapsed
    if len(set(offsets)) != 1:
        raise RuntimeError("rate layout changed between fixed-batch chunks")
    return torch.cat(chunks, dim=0), offsets[0]


def _apply_poisson_transitions(
    x: torch.Tensor,
    rates: torch.Tensor,
    offset: int,
    exposure: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    if exposure <= 0 or not math.isfinite(exposure):
        raise ValueError("Poisson exposure must be finite and positive")
    counts = torch.poisson(rates * float(exposure), generator=generator).to(torch.long)
    total_counts = counts.sum(dim=-1)
    conflict = total_counts > 1
    accepted = total_counts == 1
    sampled_reactions = int(total_counts.sum().item())
    conflict_count = int(conflict.sum().item())
    counts[conflict] = 0
    tokens = counts.argmax(dim=-1)
    updated = x.clone()
    state_view = updated[:, offset : offset + rates.shape[1]]
    if accepted.any():
        state_view[accepted] = tokens[accepted]
    return updated, {
        "accepted": int(accepted.sum().item()),
        "conflicts": conflict_count,
        "sampled_reactions": sampled_reactions,
    }


def _trapezoidal_rates_absorbing(
    q_current: torch.Tensor,
    q_intermediate: torch.Tensor,
    state_current: torch.Tensor,
    state_intermediate: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, int, float]:
    """Return positive-part theta-trapezoidal rates for one-way unmasking.

    The general method aligns reaction increments between the original and
    intermediate states. In an absorbing-mask process, positions changed by the
    predictor have no outgoing rates. Their corrected rate is zero after the
    positive-part operation. Unchanged positions retain the original coordinates.
    """
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must lie in the interval (0, 1]")
    if q_current.shape != q_intermediate.shape:
        raise RuntimeError("current and intermediate rate tensors must have equal shape")
    if state_current.shape != q_current.shape[:2] or state_intermediate.shape != state_current.shape:
        raise RuntimeError("state views do not match rate positions")
    unchanged = state_current == state_intermediate
    aligned_current = q_current * unchanged.unsqueeze(-1)
    alpha_one = 1.0 / (2.0 * theta)
    alpha_two = ((1.0 - theta) ** 2 + theta**2) / (2.0 * theta)
    raw = alpha_one * q_intermediate - alpha_two * aligned_current
    negative = raw < 0
    clipped_count = int(negative.sum().item())
    clipped_mass = float((-raw[negative]).sum().item()) if clipped_count else 0.0
    return raw.clamp_min(0.0), clipped_count, clipped_mass


def _terminal_cleanup(
    rate_function: RateFunction,
    model: Any,
    x: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
    prompt_length: int,
    eps: float,
    metrics: dict[str, Any],
    mask_id: int,
    vocab_size: int,
    temperature: float,
    cfg_scale: float,
    target_evaluation_batch_size: int,
) -> torch.Tensor:
    remaining = active_mask & (x == mask_id)
    count = int(remaining.sum().item())
    if count == 0:
        return x
    times = torch.full((x.shape[0],), eps, device=x.device, dtype=torch.float32)
    rates, offset = _evaluate_rates(
        rate_function,
        model,
        x,
        times,
        prompt_mask,
        active_mask,
        prompt_length,
        metrics,
        mask_id,
        vocab_size,
        temperature,
        cfg_scale,
        target_evaluation_batch_size,
    )
    state_view = x[:, offset : offset + rates.shape[1]]
    remaining_view = remaining[:, offset : offset + rates.shape[1]]
    tokens = rates.argmax(dim=-1)
    state_view[remaining_view] = tokens[remaining_view]
    metrics["terminal_cleanup_positions"] += count
    metrics["terminal_cleanup_rows"] += x.shape[0]
    if bool((active_mask & (x == mask_id)).any().item()):
        raise RuntimeError("terminal cleanup left a mask token in the active block")
    return x


@torch.no_grad()
def generate_approximate_ctmc(
    rate_function: RateFunction,
    model: Any,
    prompt: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    gen_length: int = 128,
    block_length: Optional[int] = None,
    steps: int = 16,
    method: str = "tau_leaping",
    theta: float = 0.5,
    mask_id: int = 126336,
    vocab_size: int = 126464,
    cfg_scale: float = 0.0,
    temperature: float = 1.0,
    eps: float = 1e-2,
    schedule: str = "loglinear",
    seed: Optional[int] = None,
    target_evaluation_batch_size: int = 4,
    return_trace: bool = False,
):
    """Sample an approximate fixed-step law using the DirectCTMC rate function.

    ``steps`` is the number of fixed intervals per block. The default trapezoidal
    parameter is theta equal to one half, matching the staged official baseline.
    """
    if method not in {"tau_leaping", "trapezoidal"}:
        raise ValueError("method must be tau_leaping or trapezoidal")
    if schedule != "loglinear":
        raise ValueError("only the staged loglinear schedule is supported")
    if vocab_size <= 1 or not 0 <= mask_id < vocab_size:
        raise ValueError("invalid vocabulary or mask token")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if cfg_scale < 0:
        raise ValueError("cfg_scale must be nonnegative")
    if method == "trapezoidal" and not 0.0 < theta <= 1.0:
        raise ValueError("theta must lie in the interval (0, 1]")

    x, prompt_mask, prompt_length = _initial_state(
        model, prompt, batch_size, gen_length, mask_id
    )
    block_length = gen_length if block_length is None else int(block_length)
    if block_length <= 0 or gen_length % block_length:
        raise ValueError("gen_length must be divisible by a positive block_length")
    grid = _time_grid(int(steps), float(eps))
    initial_seed = int(torch.seed() if seed is None else seed)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(initial_seed)
    metrics = _new_metrics()
    metrics["configured_steps_per_block"] = int(steps)
    rounds: list[dict[str, Any]] = []

    for block_index in range(gen_length // block_length):
        active_mask = _block_active_mask(
            x, prompt_length, block_index, block_length
        )
        for step_index, (time_left, time_right) in enumerate(zip(grid, grid[1:])):
            h = float(time_left - time_right)
            start_masks = int((active_mask & (x == mask_id)).sum().item())
            if start_masks == 0:
                rounds.append(
                    {
                        "block_index": block_index,
                        "step_index": step_index,
                        "start_time": time_left,
                        "end_time": time_right,
                        "start_mask_count": 0,
                        "end_mask_count": 0,
                        "skipped_no_masks": True,
                    }
                )
                metrics["completed_fixed_steps"] += 1
                continue
            times = torch.full(
                (batch_size,), time_left, device=x.device, dtype=torch.float32
            )
            q_current, offset = _evaluate_rates(
                rate_function,
                model,
                x,
                times,
                prompt_mask,
                active_mask,
                prompt_length,
                metrics,
                mask_id,
                vocab_size,
                temperature,
                cfg_scale,
                target_evaluation_batch_size,
            )
            if method == "tau_leaping":
                x, jump = _apply_poisson_transitions(
                    x, q_current, offset, h, generator
                )
                metrics["accepted_positions"] += jump["accepted"]
                metrics["conflicting_positions_discarded"] += jump["conflicts"]
                metrics["sampled_reaction_count"] += jump["sampled_reactions"]
                round_record = {
                    "block_index": block_index,
                    "step_index": step_index,
                    "start_time": time_left,
                    "end_time": time_right,
                    "start_mask_count": start_masks,
                    "accepted_positions": jump["accepted"],
                    "conflicting_positions_discarded": jump["conflicts"],
                }
            else:
                x_intermediate, predictor = _apply_poisson_transitions(
                    x, q_current, offset, theta * h, generator
                )
                intermediate_time = time_left - theta * h
                intermediate_times = torch.full(
                    (batch_size,),
                    intermediate_time,
                    device=x.device,
                    dtype=torch.float32,
                )
                q_intermediate, intermediate_offset = _evaluate_rates(
                    rate_function,
                    model,
                    x_intermediate,
                    intermediate_times,
                    prompt_mask,
                    active_mask,
                    prompt_length,
                    metrics,
                    mask_id,
                    vocab_size,
                    temperature,
                    cfg_scale,
                    target_evaluation_batch_size,
                )
                if intermediate_offset != offset:
                    raise RuntimeError("rate layout changed within a trapezoidal step")
                state_current = x[:, offset : offset + q_current.shape[1]]
                state_intermediate = x_intermediate[
                    :, offset : offset + q_current.shape[1]
                ]
                corrected_rates, clipped_count, clipped_mass = (
                    _trapezoidal_rates_absorbing(
                        q_current,
                        q_intermediate,
                        state_current,
                        state_intermediate,
                        theta,
                    )
                )
                x, corrector = _apply_poisson_transitions(
                    x_intermediate, corrected_rates, offset, h, generator
                )
                metrics["predictor_accepted_positions"] += predictor["accepted"]
                metrics["predictor_conflicting_positions_discarded"] += predictor[
                    "conflicts"
                ]
                metrics["predictor_sampled_reaction_count"] += predictor[
                    "sampled_reactions"
                ]
                metrics["accepted_positions"] += corrector["accepted"]
                metrics["conflicting_positions_discarded"] += corrector["conflicts"]
                metrics["sampled_reaction_count"] += corrector["sampled_reactions"]
                metrics["trapezoidal_clipped_entries"] += clipped_count
                metrics["trapezoidal_clipped_rate_mass"] += clipped_mass
                round_record = {
                    "block_index": block_index,
                    "step_index": step_index,
                    "start_time": time_left,
                    "intermediate_time": intermediate_time,
                    "end_time": time_right,
                    "start_mask_count": start_masks,
                    "predictor_accepted_positions": predictor["accepted"],
                    "predictor_conflicting_positions_discarded": predictor[
                        "conflicts"
                    ],
                    "corrector_accepted_positions": corrector["accepted"],
                    "corrector_conflicting_positions_discarded": corrector[
                        "conflicts"
                    ],
                    "clipped_rate_entries": clipped_count,
                    "clipped_rate_mass": clipped_mass,
                }
            end_masks = int((active_mask & (x == mask_id)).sum().item())
            if end_masks > start_masks:
                raise RuntimeError("an absorbing solver step increased the mask count")
            round_record["end_mask_count"] = end_masks
            rounds.append(round_record)
            metrics["completed_fixed_steps"] += 1

        x = _terminal_cleanup(
            rate_function,
            model,
            x,
            prompt_mask,
            active_mask,
            prompt_length,
            eps,
            metrics,
            mask_id,
            vocab_size,
            temperature,
            cfg_scale,
            target_evaluation_batch_size,
        )

    if prompt is not None:
        expected_prompt = prompt.to(x.device).expand(batch_size, -1)
        if not torch.equal(x[:, :prompt_length], expected_prompt):
            raise RuntimeError("the approximate solver changed prompt tokens")
    if bool((x[:, prompt_length:] == mask_id).any().item()):
        raise RuntimeError("the approximate solver returned a mask token")

    trace = {
        "schema_version": 1,
        "method": f"approximate_{method}",
        "target_law": "different_from_direct_ctmc",
        "exactness_claim": False,
        "initial_seed": initial_seed,
        "batch_size": batch_size,
        "gen_length": gen_length,
        "block_length": block_length,
        "steps_per_block": int(steps),
        "theta": float(theta) if method == "trapezoidal" else None,
        "eps": float(eps),
        "schedule": schedule,
        "target_evaluation_batch_size": int(target_evaluation_batch_size),
        "time_direction": "one_to_eps",
        "time_grid": grid,
        "metrics": metrics,
        "rounds": rounds,
        "output_sha256": _state_hash(x),
    }
    return (x, trace) if return_trace else x


def generate_tau_leaping(rate_function: RateFunction, model: Any, **kwargs: Any):
    kwargs.pop("method", None)
    return generate_approximate_ctmc(
        rate_function=rate_function,
        model=model,
        method="tau_leaping",
        **kwargs,
    )


def generate_trapezoidal(rate_function: RateFunction, model: Any, **kwargs: Any):
    kwargs.pop("method", None)
    return generate_approximate_ctmc(
        rate_function=rate_function,
        model=model,
        method="trapezoidal",
        **kwargs,
    )
