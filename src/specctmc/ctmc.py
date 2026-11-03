"""Fixed-batch finite-precision simulation for the LLaDA CTMC.

This additive variant preserves the mathematical target and the primary sampler.
Every target evaluation uses one fixed logical batch size. A short final batch is
padded by duplicating its last real row, and only real outputs are retained. This
defines a fixed-batch finite-precision evaluator whose numerical process is distinct
from the earlier unpadded execution when model outputs depend on batch size.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .relaxation import relaxed_acceptance_probability, validate_relaxation_value
from .projection import (
    aligned_suffix_hidden_states,
    complete_reached_probability_matrix,
    probabilities_for_hidden,
)

MASK_ID = 126336
VOCAB_SIZE = 126464
OFFICIAL_NOISE_EPS = 1e-3
DEFAULT_TARGET_EVALUATION_BATCH_SIZE = 4


def rate_noise_loglinear(t: torch.Tensor) -> torch.Tensor:
    return (1.0 - OFFICIAL_NOISE_EPS) / (
        1.0 - (1.0 - OFFICIAL_NOISE_EPS) * t
    )


def total_noise_loglinear(t: torch.Tensor) -> torch.Tensor:
    return -torch.log1p(-(1.0 - OFFICIAL_NOISE_EPS) * t)


def loglinear_rate(t: torch.Tensor) -> torch.Tensor:
    """Return the official scalar rate, algebraically equal to one divided by t."""
    return rate_noise_loglinear(t) / torch.expm1(total_noise_loglinear(t))


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prompt_cache_enabled(model: Any) -> bool:
    return bool(getattr(model, "prompt_cache_enabled", False))


def _begin_prompt_cache_request(model: Any, prompt_length: int, target_batch_size: int) -> None:
    if _prompt_cache_enabled(model):
        model.begin_request(prompt_length, target_batch_size)


def _note_prompt_cache_commits(model: Any, count: int) -> None:
    if _prompt_cache_enabled(model):
        model.note_commits(count)


def _force_prompt_cache_terminal_refresh(model: Any) -> None:
    if _prompt_cache_enabled(model):
        model.force_refresh_before_terminal_cleanup()


def _merge_prompt_cache_metrics(model: Any, metrics: Dict[str, Any]) -> None:
    if _prompt_cache_enabled(model):
        metrics.update(model.metrics())


def _prompt_cache_trace(model: Any) -> Optional[Dict[str, Any]]:
    return model.trace() if _prompt_cache_enabled(model) else None


def _preserved_sequential_distribution(model: Any, relaxation: float) -> str:
    if _prompt_cache_enabled(model):
        if relaxation == 0.0:
            return "prompt_cached_sequential_ctmc"
        return "statewise_prompt_cached_target_draft_joint_event_mixture"
    if relaxation == 0.0:
        return "released_checkpoint_full_forward_target"
    return "statewise_target_draft_joint_event_mixture"


def _state_hash(x: torch.Tensor) -> str:
    data = x.detach().to("cpu", non_blocking=False).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _validate_masks(
    x: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> None:
    if x.ndim != 2:
        raise ValueError("x must have shape [batch, sequence_length]")
    if prompt_mask.shape != x.shape:
        raise ValueError("prompt_mask must have the same shape as x")
    if active_mask.shape != x.shape:
        raise ValueError("active_mask must have the same shape as x")
    if prompt_mask.dtype != torch.bool or active_mask.dtype != torch.bool:
        raise ValueError("prompt_mask and active_mask must be boolean")


@torch.no_grad()
def llada_token_probabilities(
    model: Any, x: torch.Tensor, prompt_mask: torch.Tensor, active_mask: torch.Tensor,
    mask_id: int = MASK_ID, temperature: float = 1.0, cfg_scale: float = 0.0,
) -> torch.Tensor:
    """Return probabilities only for generated suffix positions."""
    _validate_masks(x, prompt_mask, active_mask)
    if temperature <= 0:
        raise ValueError("temperature must be strictly positive for CTMC sampling")
    if cfg_scale != 0:
        raise ValueError("suffix projection validation currently requires cfg_scale equal to zero")
    prompt_lengths = prompt_mask.sum(dim=1)
    if not torch.equal(prompt_lengths, prompt_lengths[:1].expand_as(prompt_lengths)):
        raise ValueError("all target rows must share one prompt length")
    suffix_length = x.shape[1] - int(prompt_lengths[0].item())
    if suffix_length <= 0:
        raise ValueError("generated suffix must be nonempty")
    logits = model(x, num_logits_to_keep=suffix_length).logits
    if logits.ndim != 3 or logits.shape[:2] != (x.shape[0], suffix_length):
        raise RuntimeError("suffix model logits must have shape [batch, generated_length, vocab]")
    if not 0 <= mask_id < logits.shape[-1]:
        raise ValueError("mask_id is outside the model vocabulary")
    logits = logits.float()
    logits[:, :, mask_id] = -float("inf")
    probabilities = torch.softmax(logits / float(temperature), dim=-1)
    suffix_x = x[:, -suffix_length:]
    mutable = (active_mask & (~prompt_mask) & (x == mask_id))[:, -suffix_length:]
    probabilities = probabilities * mutable.unsqueeze(-1)
    probabilities.scatter_(-1, suffix_x.unsqueeze(-1), 0.0)
    minimum_probability = torch.amin(probabilities)
    maximum_probability = torch.amax(probabilities)
    if not bool(torch.isfinite(minimum_probability).item()) or not bool(torch.isfinite(maximum_probability).item()):
        raise RuntimeError("target probabilities contain non-finite values")
    if float(minimum_probability.item()) < 0.0:
        raise RuntimeError("target probabilities contain negative values")
    row_sums = probabilities.sum(dim=-1)
    active_sums = row_sums[mutable]
    if active_sums.numel() and not torch.allclose(active_sums, torch.ones_like(active_sums), atol=2e-5, rtol=2e-5):
        raise RuntimeError("active target probability rows do not sum to one")
    inactive_sums = row_sums[~mutable]
    if inactive_sums.numel() and torch.count_nonzero(inactive_sums):
        raise RuntimeError("inactive target probability rows are not zero")
    return probabilities


@torch.no_grad()
def llada_qrate(
    model: Any,
    x: torch.Tensor,
    t: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
    mask_id: int = MASK_ID,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
) -> torch.Tensor:
    """Return the official log-linear LLaDA rate tensor."""
    _validate_masks(x, prompt_mask, active_mask)
    if t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError("t must have shape [batch]")
    if not torch.isfinite(t).all() or (t <= 0).any() or (t > 1).any():
        raise ValueError("all times must be finite and in the interval (0, 1]")

    probabilities = llada_token_probabilities(
        model=model,
        x=x,
        prompt_mask=prompt_mask,
        active_mask=active_mask,
        mask_id=mask_id,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )
    scalar_rate = loglinear_rate(t.float())
    rates = scalar_rate[:, None, None] * probabilities

    prompt_length = int(prompt_mask[0].sum().item())
    mutable = (active_mask & (~prompt_mask) & (x == mask_id))[:, prompt_length:]
    expected = mutable.sum(dim=1).float() * scalar_rate
    actual = rates.sum(dim=(1, 2))
    if not torch.allclose(actual, expected, atol=3e-3, rtol=3e-5):
        maximum_error = float((actual - expected).abs().max().item())
        raise RuntimeError(
            f"total target rate does not equal mask_count times rate, max error {maximum_error}"
        )
    return rates


def _new_metrics() -> Dict[str, Any]:
    return {
        "sequential_target_rounds": 0,
        "total_target_rate_rows": 0,
        "total_model_input_rows": 0,
        "target_padding_rows": 0,
        "target_forward_seconds": 0.0,
        "target_transformer_seconds": 0.0,
        "target_output_head_seconds": 0.0,
        "dream_alignment_seconds": 0.0,
        "target_output_head_rows_real": 0,
        "target_output_head_rows_physical": 0,
        "prompt_cache_refreshes": 0,
        "prompt_cache_full_input_rows": 0,
        "prompt_cache_suffix_input_rows": 0,
        "prompt_cache_suffix_calls": 0,
        "prompt_cache_discarded_tentative_updates": 0,
        "prompt_cache_terminal_forced_refreshes": 0,
        "prompt_cache_construction_seconds": 0.0,
        "prompt_cache_periodic_refresh_seconds": 0.0,
        "prompt_cache_total_refresh_seconds": 0.0,
        "prompt_cache_copy_seconds": 0.0,
        "prompt_cache_suffix_forward_seconds": 0.0,
        "draft_seconds": 0.0,
        "residual_seconds": 0.0,
        "bootstrap_target_rows": 0,
        "terminal_cleanup_rows": 0,
        "natural_ctmc_jumps": 0,
        "residual_corrections": 0,
        "accepted_draft_marks": 0,
        "accepted_only_due_to_relaxation": 0,
        "drafted_marks": 0,
        "speculative_rounds": 0,
        "requested_speculative_depth": None,
        "accepted_depth_histogram": {},
    }


def _target_probabilities(
    model: Any,
    states: torch.Tensor,
    times: torch.Tensor,
    prompt_masks: torch.Tensor,
    active_masks: torch.Tensor,
    metrics: Dict[str, Any],
    mask_id: int,
    temperature: float,
    cfg_scale: float,
    target_evaluation_batch_size: int,
) -> Tuple[torch.Tensor, int, float]:
    """Evaluate target rows with one fixed logical model batch size.

    Each incomplete chunk duplicates its final real row until the model receives
    ``target_evaluation_batch_size`` rows. Duplicate outputs are discarded. Target-row
    accounting counts real rows, while model-input accounting includes padding and any
    classifier-free-guidance duplication.
    """
    rows = states.shape[0]
    if rows == 0:
        prompt_length = int(prompt_masks[0].sum().item())
        empty = torch.empty(
            (0, states.shape[1] - prompt_length, 0), device=states.device, dtype=torch.float32
        )
        return empty, 0, 0.0
    size = int(target_evaluation_batch_size)
    if size <= 0:
        raise ValueError("target_evaluation_batch_size must be positive")

    chunks: List[torch.Tensor] = []
    rounds = 0
    elapsed_total = 0.0
    for start in range(0, rows, size):
        stop = min(start + size, rows)
        real_rows = stop - start
        indices = torch.arange(start, stop, device=states.device)
        padding_rows = size - real_rows
        if padding_rows:
            duplicate = torch.full(
                (padding_rows,), stop - 1, device=states.device, dtype=torch.long
            )
            indices = torch.cat((indices, duplicate), dim=0)
        evaluation_states = states.index_select(0, indices)
        evaluation_times = times.index_select(0, indices)
        evaluation_prompt_masks = prompt_masks.index_select(0, indices)
        evaluation_active_masks = active_masks.index_select(0, indices)

        _sync(states.device)
        started = time.perf_counter()
        rates = llada_qrate(
            model=model,
            x=evaluation_states,
            t=evaluation_times,
            prompt_mask=evaluation_prompt_masks,
            active_mask=evaluation_active_masks,
            mask_id=mask_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
        )
        _sync(states.device)
        elapsed = time.perf_counter() - started
        scalar = loglinear_rate(evaluation_times.float())
        probabilities = rates / scalar[:, None, None]
        chunks.append(probabilities[:real_rows])
        rounds += 1
        elapsed_total += elapsed
        metrics["sequential_target_rounds"] += 1
        metrics["total_target_rate_rows"] += real_rows
        metrics["target_padding_rows"] += padding_rows
        metrics["total_model_input_rows"] += size * (2 if cfg_scale > 0 else 1)
        metrics["target_forward_seconds"] += elapsed
    return torch.cat(chunks, dim=0), rounds, elapsed_total


def _target_selected_probabilities(
    model: Any,
    states: torch.Tensor,
    proposed_positions: Sequence[int],
    prompt_masks: torch.Tensor,
    active_masks: torch.Tensor,
    metrics: Dict[str, Any],
    mask_id: int,
    temperature: float,
    cfg_scale: float,
    target_evaluation_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """Evaluate candidate transformers once and project only proposed positions."""
    rows = states.shape[0]
    if rows != len(proposed_positions):
        raise ValueError("one proposed position is required for every target row")
    prompt_length = int(prompt_masks[0].sum().item())
    suffix_length = states.shape[1] - prompt_length
    if rows == 0:
        return (
            torch.empty((0, 0), device=states.device, dtype=torch.float32),
            torch.empty((0, suffix_length, 0), device=states.device),
            0,
            0.0,
        )
    if cfg_scale != 0.0:
        raise ValueError("selected-position projection currently requires cfg_scale equal to zero")
    size = int(target_evaluation_batch_size)
    if size <= 0:
        raise ValueError("target_evaluation_batch_size must be positive")
    probability_chunks: List[torch.Tensor] = []
    hidden_chunks: List[torch.Tensor] = []
    rounds = 0
    elapsed_total = 0.0
    for start in range(0, rows, size):
        stop = min(start + size, rows)
        real_rows = stop - start
        indices = torch.arange(start, stop, device=states.device)
        padding_rows = size - real_rows
        chunk_positions = list(int(value) for value in proposed_positions[start:stop])
        if padding_rows:
            duplicate = torch.full(
                (padding_rows,), stop - 1, device=states.device, dtype=torch.long
            )
            indices = torch.cat((indices, duplicate), dim=0)
            chunk_positions.extend([chunk_positions[-1]] * padding_rows)
        evaluation_states = states.index_select(0, indices)
        evaluation_prompt_masks = prompt_masks.index_select(0, indices)
        evaluation_active_masks = active_masks.index_select(0, indices)
        aligned_hidden, transformer_seconds, alignment_seconds = aligned_suffix_hidden_states(
            model=model,
            states=evaluation_states,
            prompt_length=prompt_length,
        )
        position_tensor = torch.tensor(
            chunk_positions, device=states.device, dtype=torch.long
        )
        row_tensor = torch.arange(size, device=states.device, dtype=torch.long)
        selected_hidden = aligned_hidden[row_tensor, position_tensor]
        selected_probabilities, head_seconds = probabilities_for_hidden(
            model=model,
            hidden=selected_hidden,
            mask_id=mask_id,
            temperature=temperature,
        )
        mutable = (
            evaluation_active_masks
            & (~evaluation_prompt_masks)
            & (evaluation_states == mask_id)
        )[:, prompt_length:]
        if not bool(mutable[row_tensor, position_tensor].all().item()):
            raise RuntimeError("a selected target position is not mutable")
        probability_chunks.append(selected_probabilities[:real_rows])
        hidden_chunks.append(aligned_hidden[:real_rows])
        elapsed = transformer_seconds + alignment_seconds + head_seconds
        rounds += 1
        elapsed_total += elapsed
        metrics["sequential_target_rounds"] += 1
        metrics["total_target_rate_rows"] += real_rows
        metrics["target_padding_rows"] += padding_rows
        metrics["total_model_input_rows"] += size
        metrics["target_forward_seconds"] += elapsed
        metrics["target_transformer_seconds"] += transformer_seconds
        metrics["target_output_head_seconds"] += head_seconds
        metrics["dream_alignment_seconds"] += alignment_seconds
        metrics["target_output_head_rows_real"] += real_rows
        metrics["target_output_head_rows_physical"] += size
    return (
        torch.cat(probability_chunks, dim=0),
        torch.cat(hidden_chunks, dim=0),
        rounds,
        elapsed_total,
    )


def _draw_uniform(generator: torch.Generator, device: torch.device) -> torch.Tensor:
    value = torch.rand((), generator=generator, device=device, dtype=torch.float64)
    return value.clamp_(torch.finfo(torch.float64).tiny, 1.0)


def _next_event_time(
    current_time: float,
    mask_count: int,
    eps: float,
    generator: torch.Generator,
    device: torch.device,
) -> Optional[float]:
    if mask_count <= 0:
        return None
    uniform = _draw_uniform(generator, device)
    exponential = -torch.log(uniform)
    next_time = current_time * math.exp(-float(exponential.item()) / mask_count)
    return next_time if next_time >= eps else None


def _sample_block_event_times(
    initial_mask_count: int,
    eps: float,
    generator: torch.Generator,
    device: torch.device,
) -> List[float]:
    """Sample the complete mark-independent event-time sequence for one block."""
    mask_count = int(initial_mask_count)
    if mask_count <= 0:
        return []
    uniforms = torch.rand(
        (mask_count,),
        generator=generator,
        device=device,
        dtype=torch.float64,
    ).clamp_(torch.finfo(torch.float64).tiny, 1.0)
    uniform_values = uniforms.to("cpu", non_blocking=False).tolist()
    current_time = 1.0
    event_times: List[float] = []
    for offset, uniform in enumerate(uniform_values):
        remaining = mask_count - offset
        exponential = -math.log(uniform)
        next_time = current_time * math.exp(-exponential / remaining)
        if next_time < eps:
            break
        event_times.append(next_time)
        current_time = next_time
    return event_times



def precompute_event_times(
    mask_count: int,
    eps: float,
    event_seed: int,
) -> List[float]:
    """Generate one block's event times on an isolated CPU random stream."""
    if mask_count < 0:
        raise ValueError("mask_count must be nonnegative")
    if not 0 < eps < 1:
        raise ValueError("eps must lie strictly between zero and one")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(event_seed))
    return _sample_block_event_times(
        mask_count, eps, generator, torch.device("cpu")
    )

def _sample_mark(
    probabilities: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[int, int]:
    """Sample a position and token jointly from equal-position mark mass."""
    if probabilities.ndim != 2:
        raise ValueError("mark probabilities must have shape [sequence_length, vocab]")
    row_mass = probabilities.sum(dim=-1)
    active_positions = torch.nonzero(row_mass > 0, as_tuple=False).flatten()
    if active_positions.numel() == 0:
        raise RuntimeError("mark probabilities have no active position")
    active_row_mass = row_mass[active_positions]
    active_offset = _sample_token(active_row_mass, generator)
    position = int(active_positions[active_offset].item())
    token = _sample_token(probabilities[position], generator)
    return position, token


def _sample_token(
    probabilities: torch.Tensor,
    generator: torch.Generator,
) -> int:
    if probabilities.ndim != 1:
        raise ValueError("categorical probabilities must be one-dimensional")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("categorical probabilities are invalid")
    total = probabilities.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0:
        raise RuntimeError("categorical probabilities have zero mass")
    normalized = probabilities / total
    return int(torch.multinomial(normalized, 1, generator=generator).item())


def _initial_state(
    model: Any,
    prompt: Optional[torch.Tensor],
    gen_length: int,
    mask_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if gen_length <= 0:
        raise ValueError("gen_length must be positive")
    device = _model_device(model, prompt)
    if prompt is not None:
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("prompt must have shape [1, prompt_length]")
        prompt_length = int(prompt.shape[1])
    else:
        prompt_length = 0
    x = torch.full(
        (1, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    if prompt is not None:
        x[:, :prompt_length] = prompt.to(device)
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
    stop = start + block_length
    active[:, start:stop] = True
    return active


def _terminal_cleanup(
    model: Any,
    x: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
    eps: float,
    metrics: Dict[str, Any],
    mask_id: int,
    temperature: float,
    cfg_scale: float,
    target_evaluation_batch_size: int,
) -> int:
    remaining = active_mask & (x == mask_id)
    count = int(remaining.sum().item())
    if count == 0:
        return 0
    times = torch.full((1,), eps, device=x.device, dtype=torch.float32)
    probabilities, _, _ = _target_probabilities(
        model=model,
        states=x,
        times=times,
        prompt_masks=prompt_mask,
        active_masks=active_mask,
        metrics=metrics,
        mask_id=mask_id,
        temperature=temperature,
        cfg_scale=cfg_scale,
        target_evaluation_batch_size=target_evaluation_batch_size,
    )
    prompt_length = int(prompt_mask[0].sum().item())
    remaining_suffix = remaining[:, prompt_length:]
    x[:, prompt_length:][remaining_suffix] = probabilities.argmax(dim=-1)[remaining_suffix]
    metrics["terminal_cleanup_rows"] += 1
    return count


def _aggregate_request_metrics(requests: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "sequential_target_rounds",
        "total_target_rate_rows",
        "total_model_input_rows",
        "target_padding_rows",
        "target_forward_seconds",
        "target_transformer_seconds",
        "target_output_head_seconds",
        "dream_alignment_seconds",
        "target_output_head_rows_real",
        "target_output_head_rows_physical",
        "prompt_cache_refreshes",
        "prompt_cache_full_input_rows",
        "prompt_cache_suffix_input_rows",
        "prompt_cache_suffix_calls",
        "prompt_cache_discarded_tentative_updates",
        "prompt_cache_terminal_forced_refreshes",
        "prompt_cache_construction_seconds",
        "prompt_cache_periodic_refresh_seconds",
        "prompt_cache_total_refresh_seconds",
        "prompt_cache_copy_seconds",
        "prompt_cache_suffix_forward_seconds",
        "draft_seconds",
        "residual_seconds",
        "bootstrap_target_rows",
        "terminal_cleanup_rows",
        "natural_ctmc_jumps",
        "residual_corrections",
        "accepted_draft_marks",
        "accepted_only_due_to_relaxation",
        "drafted_marks",
        "speculative_rounds",
    ]
    output: Dict[str, Any] = {}
    for key in keys:
        output[key] = sum(request["metrics"][key] for request in requests)
    histogram: Counter[int] = Counter()
    for request in requests:
        histogram.update(
            {
                int(depth): int(count)
                for depth, count in request["metrics"][
                    "accepted_depth_histogram"
                ].items()
            }
        )
    output["accepted_depth_histogram"] = {
        str(depth): histogram[depth] for depth in sorted(histogram)
    }
    output["mean_accepted_depth_per_speculative_round"] = (
        output["accepted_draft_marks"] / output["speculative_rounds"]
        if output["speculative_rounds"]
        else None
    )
    return output


@torch.no_grad()
def _generate_exact_single(
    model: Any,
    prompt: Optional[torch.Tensor],
    gen_length: int,
    block_length: int,
    mask_id: int,
    vocab_size: int,
    cfg_scale: float,
    temperature: float,
    eps: float,
    seed: int,
    event_times_by_block: Optional[Sequence[Sequence[float]]],
    stream_seeds: Optional[Dict[str, int]],
    target_evaluation_batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    x, prompt_mask, prompt_length = _initial_state(
        model, prompt, gen_length, mask_id
    )
    _begin_prompt_cache_request(model, prompt_length, target_evaluation_batch_size)
    if not 0 < eps < 1:
        raise ValueError("eps must lie strictly between zero and one")
    if vocab_size <= 1 or not 0 <= mask_id < vocab_size:
        raise ValueError("invalid vocabulary or mask token")
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")

    seeds = {"event": int(seed), "target": int(seed) + 1000003}
    if stream_seeds is not None:
        seeds.update({key: int(value) for key, value in stream_seeds.items()})
    event_generator = torch.Generator(device=x.device)
    event_generator.manual_seed(seeds["event"])
    target_generator = torch.Generator(device=x.device)
    target_generator.manual_seed(seeds["target"])
    metrics = _new_metrics()
    rounds: List[Dict[str, Any]] = []
    event_times: List[float] = []

    for block_index in range(gen_length // block_length):
        active_mask = _block_active_mask(
            x, prompt_length, block_index, block_length
        )
        initial_mask_count = int((active_mask & (x == mask_id)).sum().item())
        if event_times_by_block is None:
            block_event_times = _sample_block_event_times(
                initial_mask_count, eps, event_generator, x.device
            )
        else:
            if len(event_times_by_block) != gen_length // block_length:
                raise ValueError("event_times_by_block has the wrong block count")
            block_event_times = [
                float(value) for value in event_times_by_block[block_index]
            ]
            if len(block_event_times) > initial_mask_count:
                raise ValueError("too many precomputed event times")
            if any(
                left <= right or right < eps
                for left, right in zip([1.0] + block_event_times, block_event_times)
            ):
                raise ValueError("precomputed event times are not strictly descending")
        previous_time = 1.0
        for next_time in block_event_times:
            mutable = active_mask & (x == mask_id)
            mask_count = int(mutable.sum().item())
            time_tensor = torch.tensor(
                [next_time], device=x.device, dtype=torch.float32
            )
            probabilities, microbatches, forward_seconds = _target_probabilities(
                model=model,
                states=x,
                times=time_tensor,
                prompt_masks=prompt_mask,
                active_masks=active_mask,
                metrics=metrics,
                mask_id=mask_id,
                temperature=temperature,
                cfg_scale=cfg_scale,
                target_evaluation_batch_size=target_evaluation_batch_size,
            )
            relative_position, token = _sample_mark(probabilities[0], target_generator)
            position = prompt_length + relative_position
            if not bool(mutable[0, position].item()) or token == mask_id:
                raise RuntimeError("a natural CTMC jump selected an invalid mark")
            x[0, position] = token
            if int((active_mask & (x == mask_id)).sum().item()) != mask_count - 1:
                raise RuntimeError("a CTMC jump did not reduce the mask count by one")
            metrics["natural_ctmc_jumps"] += 1
            _note_prompt_cache_commits(model, 1)
            event_times.append(next_time)
            rounds.append(
                {
                    "round_index": len(rounds),
                    "block_index": block_index,
                    "start_time": previous_time,
                    "event_times": [next_time],
                    "start_mask_count": mask_count,
                    "end_mask_count": mask_count - 1,
                    "selected_position": position,
                    "selected_token": token,
                    "target_rate_rows": 1,
                    "target_evaluation_calls": microbatches,
                    "target_forward_seconds": forward_seconds,
                }
            )
            previous_time = next_time
        _force_prompt_cache_terminal_refresh(model)
        cleanup_count = _terminal_cleanup(
            model=model,
            x=x,
            prompt_mask=prompt_mask,
            active_mask=active_mask,
            eps=eps,
            metrics=metrics,
            mask_id=mask_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
            target_evaluation_batch_size=target_evaluation_batch_size,
        )
        if cleanup_count:
            rounds.append(
                {
                    "round_index": len(rounds),
                    "block_index": block_index,
                    "kind": "terminal_cleanup",
                    "cleanup_mask_count": cleanup_count,
                    "target_rate_rows": 1,
                }
            )

    _merge_prompt_cache_metrics(model, metrics)
    request = {
        "seed": int(seed),
        "stream_seeds": seeds,
        "precomputed_event_times": event_times_by_block is not None,
        "method": "ctmc_exact",
        "cache_mode": getattr(model, "cache_mode", "none"),
        "prompt_cache_trace": _prompt_cache_trace(model),
        "preserved_sequential_distribution": _preserved_sequential_distribution(model, 0.0),
        "event_times": event_times,
        "rounds": rounds,
        "metrics": metrics,
        "output_sha256": _state_hash(x),
    }
    return x, request


@torch.no_grad()
def generate_ctmc_exact(
    model: Any,
    prompt: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    gen_length: int = 128,
    block_length: Optional[int] = None,
    mask_id: int = MASK_ID,
    vocab_size: int = VOCAB_SIZE,
    cfg_scale: float = 0.0,
    temperature: float = 1.0,
    eps: float = 1e-2,
    schedule: str = "loglinear",
    seed: Optional[int] = None,
    event_times_by_block: Optional[Sequence[Sequence[float]]] = None,
    stream_seeds: Optional[Dict[str, int]] = None,
    target_evaluation_batch_size: int = DEFAULT_TARGET_EVALUATION_BATCH_SIZE,
    return_trace: bool = False,
):
    """Sample the event-driven CTMC with the fixed-batch numerical evaluator."""
    if schedule != "loglinear":
        raise ValueError("only the verified official loglinear schedule is supported")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if target_evaluation_batch_size <= 0:
        raise ValueError("target_evaluation_batch_size must be positive")
    block_length = gen_length if block_length is None else int(block_length)
    initial_seed = int(torch.seed() if seed is None else seed)
    outputs: List[torch.Tensor] = []
    requests: List[Dict[str, Any]] = []
    for request_index in range(batch_size):
        request_stream_seeds = (
            {key: int(value) + request_index for key, value in stream_seeds.items()}
            if stream_seeds is not None
            else None
        )
        output, request = _generate_exact_single(
            model=model,
            prompt=prompt,
            gen_length=gen_length,
            block_length=block_length,
            mask_id=mask_id,
            vocab_size=vocab_size,
            cfg_scale=cfg_scale,
            temperature=temperature,
            eps=eps,
            seed=initial_seed + request_index,
            event_times_by_block=event_times_by_block,
            stream_seeds=request_stream_seeds,
            target_evaluation_batch_size=target_evaluation_batch_size,
        )
        request["request_index"] = request_index
        outputs.append(output)
        requests.append(request)
    result = torch.cat(outputs, dim=0)
    trace = {
        "schema_version": 1,
        "method": "ctmc_exact_fixed_batch",
        "batch_size": batch_size,
        "gen_length": gen_length,
        "block_length": block_length,
        "eps": eps,
        "temperature": temperature,
        "cfg_scale": cfg_scale,
        "initial_seed": initial_seed,
        "target_evaluation_batch_size": target_evaluation_batch_size,
        "finite_precision_evaluator": "fixed_batch_duplicate_padding",
        "cache_mode": getattr(model, "cache_mode", "none"),
        "prompt_cache_refresh_interval": getattr(model, "prompt_cache_refresh_interval", None),
        "preserved_sequential_distribution": _preserved_sequential_distribution(model, 0.0),
        "random_draw_order": [
            "all_block_event_clock_uniforms",
            "position_uniform_per_event",
            "target_token_categorical_per_event",
        ],
        "requests": requests,
        "aggregate": _aggregate_request_metrics(requests),
    }
    return (result, trace) if return_trace else result


def _uniform_nonmask(
    rows: int,
    vocab_size: int,
    mask_id: int,
    device: torch.device,
) -> torch.Tensor:
    probabilities = torch.full(
        (rows, vocab_size),
        1.0 / float(vocab_size - 1),
        device=device,
        dtype=torch.float32,
    )
    probabilities[:, mask_id] = 0.0
    return probabilities


def _draft_probabilities(
    x: torch.Tensor, active_mask: torch.Tensor, prompt_length: int,
    stale_cache: Optional[torch.Tensor], vocab_size: int, mask_id: int,
    uniform_mix: float, exact_cache_for_state: bool,
) -> torch.Tensor:
    if not 0 <= uniform_mix < 1:
        raise ValueError("draft_uniform_mix must lie in [0, 1)")
    mutable = active_mask[0, prompt_length:] & (x[0, prompt_length:] == mask_id)
    positions = torch.nonzero(mutable, as_tuple=False).flatten()
    output = torch.zeros((x.shape[1] - prompt_length, vocab_size), device=x.device, dtype=torch.float32)
    if positions.numel() == 0:
        return output
    uniform = _uniform_nonmask(int(positions.numel()), vocab_size, mask_id, x.device)
    if stale_cache is None:
        rows = uniform
    else:
        rows = stale_cache[positions].float().clone()
        rows[:, mask_id] = 0.0
        sums = rows.sum(dim=-1, keepdim=True)
        invalid = (~torch.isfinite(sums)) | (sums <= 0)
        if invalid.any():
            rows[invalid.flatten()] = uniform[invalid.flatten()]
            sums = rows.sum(dim=-1, keepdim=True)
        rows = rows / sums
        if uniform_mix and not exact_cache_for_state:
            rows = (1.0 - uniform_mix) * rows + uniform_mix * uniform
    output[positions] = rows
    active_sums = output[positions].sum(dim=-1)
    if not torch.allclose(active_sums, torch.ones_like(active_sums), atol=2e-5, rtol=2e-5):
        raise RuntimeError("draft probability rows do not sum to one")
    return output


def _sample_residual(
    target: torch.Tensor,
    draft: torch.Tensor,
    mutable: torch.Tensor,
    generator: torch.Generator,
    mask_id: int,
) -> Tuple[int, int, float]:
    started = time.perf_counter()
    residual = (target - draft).clamp_min(0.0)
    residual[~mutable] = 0.0
    residual[:, mask_id] = 0.0
    active_positions = torch.nonzero(
        residual.sum(dim=-1) > 0, as_tuple=False
    ).flatten()
    if active_positions.numel() == 0:
        raise RuntimeError(
            "a rejected proposal had zero positive residual mass"
        )
    active_residual = residual[active_positions]
    active_row_mass = active_residual.sum(dim=-1)
    total = active_row_mass.sum()
    if not torch.isfinite(total) or float(total.item()) <= 1e-8:
        raise RuntimeError(
            "a rejected proposal had zero positive residual mass"
        )
    active_offset = _sample_token(active_row_mass, generator)
    position = int(active_positions[active_offset].item())
    token = _sample_token(residual[position], generator)
    if not bool(mutable[position].item()) or token == mask_id:
        raise RuntimeError("residual correction selected an invalid mark")
    return position, token, time.perf_counter() - started


@torch.no_grad()
def _generate_speculative_single(
    model: Any,
    prompt: Optional[torch.Tensor],
    gen_length: int,
    block_length: int,
    speculative_depth: int,
    target_evaluation_batch_size: int,
    draft_kind: str,
    draft_uniform_mix: float,
    bootstrap_kind: str,
    mask_id: int,
    vocab_size: int,
    cfg_scale: float,
    temperature: float,
    eps: float,
    seed: int,
    event_times_by_block: Optional[Sequence[Sequence[float]]],
    stream_seeds: Optional[Dict[str, int]],
    selected_position_projection: bool,
    verification_mixture_lambda: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if speculative_depth <= 0:
        raise ValueError("speculative_depth must be positive")
    verification_mixture_lambda = validate_relaxation_value(verification_mixture_lambda)
    if bootstrap_kind not in {"uniform", "target"}:
        raise ValueError("bootstrap_kind must be uniform or target")
    if draft_kind not in {"stale_logits", "uniform"}:
        raise ValueError("draft_kind must be stale_logits or uniform")
    if draft_kind == "uniform" and bootstrap_kind != "uniform":
        raise ValueError("uniform drafting requires uniform bootstrap")
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")

    x, prompt_mask, prompt_length = _initial_state(
        model, prompt, gen_length, mask_id
    )
    _begin_prompt_cache_request(model, prompt_length, target_evaluation_batch_size)
    seeds = {
        "event": int(seed),
        "draft": int(seed) + 2000003,
        "verification": int(seed) + 3000017,
        "residual": int(seed) + 4000037,
    }
    if stream_seeds is not None:
        seeds.update({key: int(value) for key, value in stream_seeds.items()})
    event_generator = torch.Generator(device=x.device)
    event_generator.manual_seed(seeds["event"])
    draft_generator = torch.Generator(device=x.device)
    draft_generator.manual_seed(seeds["draft"])
    verification_generator = torch.Generator(device=x.device)
    verification_generator.manual_seed(seeds["verification"])
    residual_generator = torch.Generator(device=x.device)
    residual_generator.manual_seed(seeds["residual"])
    metrics = _new_metrics()
    metrics["requested_speculative_depth"] = speculative_depth
    rounds: List[Dict[str, Any]] = []
    event_times: List[float] = []
    depth_histogram: Counter[int] = Counter()

    for block_index in range(gen_length // block_length):
        active_mask = _block_active_mask(
            x, prompt_length, block_index, block_length
        )
        initial_mask_count = int((active_mask & (x == mask_id)).sum().item())
        if event_times_by_block is None:
            block_event_times = _sample_block_event_times(
                initial_mask_count, eps, event_generator, x.device
            )
        else:
            if len(event_times_by_block) != gen_length // block_length:
                raise ValueError("event_times_by_block has the wrong block count")
            block_event_times = [
                float(value) for value in event_times_by_block[block_index]
            ]
            if len(block_event_times) > initial_mask_count:
                raise ValueError("too many precomputed event times")
            if any(
                left <= right or right < eps
                for left, right in zip([1.0] + block_event_times, block_event_times)
            ):
                raise ValueError("precomputed event times are not strictly descending")
        event_cursor = 0
        stale_cache: Optional[torch.Tensor] = None
        cache_is_exact_for_current_state = False
        bootstrap_rows_pending = 0

        if bootstrap_kind == "target" and bool(
            (active_mask & (x == mask_id)).any().item()
        ):
            bootstrap_time = torch.tensor(
                [1.0], device=x.device, dtype=torch.float32
            )
            bootstrap_probabilities, _, _ = _target_probabilities(
                model=model,
                states=x,
                times=bootstrap_time,
                prompt_masks=prompt_mask,
                active_masks=active_mask,
                metrics=metrics,
                mask_id=mask_id,
                temperature=temperature,
                cfg_scale=cfg_scale,
                target_evaluation_batch_size=target_evaluation_batch_size,
            )
            stale_cache = bootstrap_probabilities[0].detach().clone()
            cache_is_exact_for_current_state = True
            metrics["bootstrap_target_rows"] += 1
            bootstrap_rows_pending = 1

        while event_cursor < len(block_event_times):
            mutable = active_mask & (x == mask_id)
            start_mask_count = int(mutable.sum().item())
            cache_remaining_before_round = (
                model.remaining_before_refresh()
                if _prompt_cache_enabled(model)
                else speculative_depth
            )
            effective_speculative_depth = min(speculative_depth, cache_remaining_before_round)
            round_event_times = block_event_times[
                event_cursor : event_cursor + effective_speculative_depth
            ]
            round_start_time = (
                1.0 if event_cursor == 0 else block_event_times[event_cursor - 1]
            )

            draft_started = time.perf_counter()
            draft_state = x.clone()
            draft_prestates: List[torch.Tensor] = []
            draft_times: List[float] = []
            draft_positions: List[int] = []
            draft_tokens: List[int] = []
            draft_proposal_probabilities: List[float] = []
            verification_uniforms: List[float] = []
            draft_matrices: List[torch.Tensor] = []

            for draft_index, next_time in enumerate(round_event_times):
                draft_mutable = active_mask & (draft_state == mask_id)
                exact_cache = (
                    draft_index == 0 and cache_is_exact_for_current_state
                )
                draft_probability = _draft_probabilities(
                    x=draft_state,
                    active_mask=active_mask,
                    prompt_length=prompt_length,
                    stale_cache=None if draft_kind == "uniform" else stale_cache,
                    vocab_size=vocab_size,
                    mask_id=mask_id,
                    uniform_mix=0.0 if draft_kind == "uniform" else draft_uniform_mix,
                    exact_cache_for_state=False if draft_kind == "uniform" else exact_cache,
                )
                position, token = _sample_mark(draft_probability, draft_generator)
                if not bool(draft_mutable[0, prompt_length + position].item()) or token == mask_id:
                    raise RuntimeError("the draft selected an invalid mark")
                proposal_probability = float(
                    draft_probability[position, token].item()
                )
                verification_uniform = float(
                    _draw_uniform(verification_generator, x.device).item()
                )

                draft_prestates.append(draft_state.clone())
                draft_times.append(float(next_time))
                draft_positions.append(position)
                draft_tokens.append(token)
                draft_proposal_probabilities.append(proposal_probability)
                verification_uniforms.append(verification_uniform)
                draft_matrices.append(draft_probability)
                draft_state[0, prompt_length + position] = token

            draft_seconds = time.perf_counter() - draft_started
            metrics["draft_seconds"] += draft_seconds
            realized_depth = len(draft_prestates)
            if realized_depth == 0:
                raise RuntimeError("event drafting produced no proposals")
            metrics["drafted_marks"] += realized_depth
            metrics["speculative_rounds"] += 1

            reuse_first = cache_is_exact_for_current_state
            target_start = 1 if reuse_first else 0
            target_selected_probabilities: Optional[torch.Tensor] = None
            target_hidden_states: Optional[torch.Tensor] = None
            target_probabilities: Optional[torch.Tensor] = None
            if target_start < realized_depth:
                stacked_states = torch.cat(
                    draft_prestates[target_start:], dim=0
                )
                stacked_times = torch.tensor(
                    draft_times[target_start:],
                    device=x.device,
                    dtype=torch.float32,
                )
                stacked_prompt_masks = prompt_mask.expand(
                    realized_depth - target_start, -1
                )
                stacked_active_masks = active_mask.expand(
                    realized_depth - target_start, -1
                )
                if selected_position_projection:
                    (
                        target_selected_probabilities,
                        target_hidden_states,
                        target_microbatches,
                        target_seconds,
                    ) = _target_selected_probabilities(
                        model=model,
                        states=stacked_states,
                        proposed_positions=draft_positions[target_start:],
                        prompt_masks=stacked_prompt_masks,
                        active_masks=stacked_active_masks,
                        metrics=metrics,
                        mask_id=mask_id,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        target_evaluation_batch_size=target_evaluation_batch_size,
                    )
                else:
                    target_probabilities, target_microbatches, target_seconds = (
                        _target_probabilities(
                            model=model,
                            states=stacked_states,
                            times=stacked_times,
                            prompt_masks=stacked_prompt_masks,
                            active_masks=stacked_active_masks,
                            metrics=metrics,
                            mask_id=mask_id,
                            temperature=temperature,
                            cfg_scale=cfg_scale,
                            target_evaluation_batch_size=target_evaluation_batch_size,
                        )
                    )
            else:
                target_microbatches = 0
                target_seconds = 0.0

            accepted = 0
            rejected = False
            correction_position: Optional[int] = None
            correction_token: Optional[int] = None
            acceptance_probabilities: List[float] = []
            exact_acceptance_probabilities: List[float] = []
            proposal_accepted: List[bool] = []
            accepted_only_due_to_relaxation: List[bool] = []
            committed_times: List[float] = []
            latest_target_probability: Optional[torch.Tensor] = None
            latest_selected_probability: Optional[torch.Tensor] = None
            latest_target_hidden: Optional[torch.Tensor] = None
            latest_target_prestate: Optional[torch.Tensor] = None
            latest_target_position: Optional[int] = None

            for proposal_index in range(realized_depth):
                prestate = draft_prestates[proposal_index]
                if not torch.equal(x, prestate):
                    raise RuntimeError(
                        "a verified draft prestate does not match the committed state"
                    )
                position = draft_positions[proposal_index]
                token = draft_tokens[proposal_index]
                draft_probability = draft_matrices[proposal_index]
                target_probability: Optional[torch.Tensor]
                target_hidden: Optional[torch.Tensor] = None
                if proposal_index < target_start:
                    if stale_cache is None:
                        raise RuntimeError("target bootstrap cache is absent")
                    target_probability = stale_cache
                    target_selected_probability = target_probability[position]
                elif selected_position_projection:
                    if target_selected_probabilities is None or target_hidden_states is None:
                        raise RuntimeError("selected target evaluation is absent")
                    selected_index = proposal_index - target_start
                    target_probability = None
                    target_selected_probability = target_selected_probabilities[selected_index]
                    target_hidden = target_hidden_states[selected_index]
                else:
                    if target_probabilities is None:
                        raise RuntimeError("complete target evaluation is absent")
                    target_probability = target_probabilities[proposal_index - target_start]
                    target_selected_probability = target_probability[position]

                recorded = draft_proposal_probabilities[proposal_index]
                used = float(draft_probability[position, token].item())
                if not math.isclose(recorded, used, rel_tol=0.0, abs_tol=1e-8):
                    raise RuntimeError(
                        "recorded draft probability changed before verification"
                    )
                target_value = float(target_selected_probability[token].item())
                if used <= 0:
                    raise RuntimeError("draft proposal had zero probability")
                exact_accept_probability = min(1.0, target_value / used)
                accept_probability = relaxed_acceptance_probability(
                    target_probability=target_value,
                    draft_probability=used,
                    verification_mixture_lambda=verification_mixture_lambda,
                )
                exact_acceptance_probabilities.append(exact_accept_probability)
                acceptance_probabilities.append(accept_probability)

                accept = (
                    proposal_index < target_start
                    or verification_uniforms[proposal_index] < accept_probability
                )
                exact_accept = (
                    proposal_index < target_start
                    or verification_uniforms[proposal_index] < exact_accept_probability
                )
                relaxed_only = bool(accept and not exact_accept)
                proposal_accepted.append(bool(accept))
                accepted_only_due_to_relaxation.append(relaxed_only)
                if relaxed_only:
                    metrics["accepted_only_due_to_relaxation"] += 1
                latest_target_probability = target_probability
                latest_selected_probability = target_selected_probability
                latest_target_hidden = target_hidden
                latest_target_prestate = prestate
                latest_target_position = position
                cache_is_exact_for_current_state = False
                if accept:
                    x[0, prompt_length + position] = token
                    accepted += 1
                    metrics["accepted_draft_marks"] += 1
                    metrics["natural_ctmc_jumps"] += 1
                    committed_times.append(draft_times[proposal_index])
                    continue

                residual_mutable = (active_mask[0] & (prestate[0] == mask_id))[prompt_length:]
                if target_probability is None:
                    if target_hidden is None:
                        raise RuntimeError("selected target hidden state is absent")
                    target_probability, reached_head_seconds, reached_rows = (
                        complete_reached_probability_matrix(
                            model=model,
                            aligned_hidden=target_hidden,
                            mutable=residual_mutable,
                            mask_id=mask_id,
                            temperature=temperature,
                            selected_position=position,
                            selected_probability=target_selected_probability,
                        )
                    )
                    metrics["target_output_head_seconds"] += reached_head_seconds
                    metrics["target_forward_seconds"] += reached_head_seconds
                    metrics["target_output_head_rows_real"] += reached_rows
                    metrics["target_output_head_rows_physical"] += reached_rows
                    latest_target_probability = target_probability
                correction_position, correction_token, residual_seconds = (
                    _sample_residual(
                        target=target_probability,
                        draft=draft_probability,
                        mutable=residual_mutable,
                        generator=residual_generator,
                        mask_id=mask_id,
                    )
                )
                metrics["residual_seconds"] += residual_seconds
                x[0, prompt_length + correction_position] = correction_token
                metrics["residual_corrections"] += 1
                metrics["natural_ctmc_jumps"] += 1
                committed_times.append(draft_times[proposal_index])
                rejected = True
                break

            if latest_target_probability is None and selected_position_projection:
                if (
                    latest_target_hidden is None
                    or latest_selected_probability is None
                    or latest_target_prestate is None
                    or latest_target_position is None
                ):
                    raise RuntimeError("verification produced no selected target state")
                reached_mutable = (
                    active_mask[0] & (latest_target_prestate[0] == mask_id)
                )[prompt_length:]
                latest_target_probability, reached_head_seconds, reached_rows = (
                    complete_reached_probability_matrix(
                        model=model,
                        aligned_hidden=latest_target_hidden,
                        mutable=reached_mutable,
                        mask_id=mask_id,
                        temperature=temperature,
                        selected_position=latest_target_position,
                        selected_probability=latest_selected_probability,
                    )
                )
                metrics["target_output_head_seconds"] += reached_head_seconds
                metrics["target_forward_seconds"] += reached_head_seconds
                metrics["target_output_head_rows_real"] += reached_rows
                metrics["target_output_head_rows_physical"] += reached_rows
            if latest_target_probability is None:
                raise RuntimeError("verification produced no target probability")
            stale_cache = latest_target_probability.detach().clone()
            committed_count = accepted + int(rejected)
            end_mask_count = int((active_mask & (x == mask_id)).sum().item())
            if end_mask_count != start_mask_count - committed_count:
                raise RuntimeError(
                    "committed speculative marks did not reduce mask count correctly"
                )
            _note_prompt_cache_commits(model, committed_count)
            event_cursor += committed_count
            event_times.extend(committed_times)
            depth_histogram[accepted] += 1
            metrics["accepted_depth_histogram"] = {
                str(depth): depth_histogram[depth]
                for depth in sorted(depth_histogram)
            }
            rounds.append(
                {
                    "round_index": len(rounds),
                    "block_index": block_index,
                    "start_time": round_start_time,
                    "drafted_event_times": draft_times,
                    "committed_event_times": committed_times,
                    "start_mask_count": start_mask_count,
                    "end_mask_count": end_mask_count,
                    "requested_depth": speculative_depth,
                    "effective_depth_before_cache_boundary": effective_speculative_depth,
                    "cache_remaining_before_round": cache_remaining_before_round,
                    "realized_depth": realized_depth,
                    "accepted_draft_depth": accepted,
                    "residual_correction": rejected,
                    "corrected_position": (None if correction_position is None else prompt_length + correction_position),
                    "corrected_token": correction_token,
                    "bootstrap_target_rows": bootstrap_rows_pending,
                    "reused_bootstrap_first_row": reuse_first,
                    "target_rate_rows": realized_depth - target_start,
                    "target_evaluation_calls": target_microbatches,
                    "target_forward_seconds": target_seconds,
                    "draft_seconds": draft_seconds,
                    "exact_acceptance_probabilities": exact_acceptance_probabilities,
                    "acceptance_probabilities": acceptance_probabilities,
                    "verification_uniforms": verification_uniforms[: len(acceptance_probabilities)],
                    "proposal_accepted": proposal_accepted,
                    "accepted_only_due_to_relaxation": accepted_only_due_to_relaxation,
                    "minimum_acceptance_probability": min(
                        acceptance_probabilities
                    ),
                    "mean_acceptance_probability": sum(
                        acceptance_probabilities
                    )
                    / len(acceptance_probabilities),
                }
            )
            bootstrap_rows_pending = 0

        _force_prompt_cache_terminal_refresh(model)
        cleanup_count = _terminal_cleanup(
            model=model,
            x=x,
            prompt_mask=prompt_mask,
            active_mask=active_mask,
            eps=eps,
            metrics=metrics,
            mask_id=mask_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
            target_evaluation_batch_size=target_evaluation_batch_size,
        )
        if cleanup_count:
            rounds.append(
                {
                    "round_index": len(rounds),
                    "block_index": block_index,
                    "kind": "terminal_cleanup",
                    "cleanup_mask_count": cleanup_count,
                    "target_rate_rows": 1,
                }
            )

    _merge_prompt_cache_metrics(model, metrics)
    request = {
        "seed": int(seed),
        "stream_seeds": seeds,
        "precomputed_event_times": event_times_by_block is not None,
        "method": "ctmc_speculative_fixed_batch",
        "cache_mode": getattr(model, "cache_mode", "none"),
        "prompt_cache_trace": _prompt_cache_trace(model),
        "speculative_depth": speculative_depth,
        "selected_position_output_projection": selected_position_projection,
        "verification_mixture_lambda": verification_mixture_lambda,
        "preserved_sequential_distribution": _preserved_sequential_distribution(
            model, verification_mixture_lambda
        ),
        "event_times": event_times,
        "rounds": rounds,
        "metrics": metrics,
        "output_sha256": _state_hash(x),
    }
    return x, request


@torch.no_grad()
def generate_ctmc_speculative(
    model: Any,
    prompt: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    gen_length: int = 128,
    block_length: Optional[int] = None,
    speculative_depth: int = 4,
    target_evaluation_batch_size: int = DEFAULT_TARGET_EVALUATION_BATCH_SIZE,
    draft_kind: str = "stale_logits",
    draft_uniform_mix: float = 1e-4,
    bootstrap_kind: str = "uniform",
    mask_id: int = MASK_ID,
    vocab_size: int = VOCAB_SIZE,
    cfg_scale: float = 0.0,
    temperature: float = 1.0,
    eps: float = 1e-2,
    schedule: str = "loglinear",
    seed: Optional[int] = None,
    event_times_by_block: Optional[Sequence[Sequence[float]]] = None,
    stream_seeds: Optional[Dict[str, int]] = None,
    return_trace: bool = False,
    selected_position_projection: bool = False,
    verification_mixture_lambda: float = 0.0,
):
    """Sample speculatively with the fixed-batch numerical evaluator."""
    if schedule != "loglinear":
        raise ValueError("only the verified official loglinear schedule is supported")
    if draft_kind not in {"stale_logits", "uniform"}:
        raise ValueError("draft_kind must be stale_logits or uniform")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if target_evaluation_batch_size <= 0:
        raise ValueError("target_evaluation_batch_size must be positive")
    verification_mixture_lambda = validate_relaxation_value(verification_mixture_lambda)
    if _prompt_cache_enabled(model) and selected_position_projection:
        raise ValueError("prompt cache and selected-position projection require a separate composition test")
    block_length = gen_length if block_length is None else int(block_length)
    initial_seed = int(torch.seed() if seed is None else seed)
    outputs: List[torch.Tensor] = []
    requests: List[Dict[str, Any]] = []
    for request_index in range(batch_size):
        request_stream_seeds = (
            {key: int(value) + request_index for key, value in stream_seeds.items()}
            if stream_seeds is not None
            else None
        )
        output, request = _generate_speculative_single(
            model=model,
            prompt=prompt,
            gen_length=gen_length,
            block_length=block_length,
            speculative_depth=speculative_depth,
            target_evaluation_batch_size=target_evaluation_batch_size,
            draft_kind=draft_kind,
            draft_uniform_mix=draft_uniform_mix,
            bootstrap_kind=bootstrap_kind,
            mask_id=mask_id,
            vocab_size=vocab_size,
            cfg_scale=cfg_scale,
            temperature=temperature,
            eps=eps,
            seed=initial_seed + request_index,
            event_times_by_block=event_times_by_block,
            stream_seeds=request_stream_seeds,
            selected_position_projection=selected_position_projection,
            verification_mixture_lambda=verification_mixture_lambda,
        )
        request["request_index"] = request_index
        outputs.append(output)
        requests.append(request)
    result = torch.cat(outputs, dim=0)
    trace = {
        "schema_version": 2,
        "method": "ctmc_speculative",
        "batch_size": batch_size,
        "gen_length": gen_length,
        "block_length": block_length,
        "eps": eps,
        "temperature": temperature,
        "cfg_scale": cfg_scale,
        "initial_seed": initial_seed,
        "speculative_depth": speculative_depth,
        "target_evaluation_batch_size": target_evaluation_batch_size,
        "finite_precision_evaluator": (
            "fixed_batch_duplicate_padding_selected_position_projection"
            if selected_position_projection
            else "fixed_batch_duplicate_padding"
        ),
        "selected_position_output_projection": selected_position_projection,
        "cache_mode": getattr(model, "cache_mode", "none"),
        "prompt_cache_refresh_interval": getattr(model, "prompt_cache_refresh_interval", None),
        "verification_distribution": "statewise_target_draft_joint_event_mixture",
        "verification_mixture_lambda": verification_mixture_lambda,
        "residual_distribution": "normalized_positive_target_minus_draft",
        "preserved_sequential_distribution": _preserved_sequential_distribution(
            model, verification_mixture_lambda
        ),
        "draft_kind": draft_kind,
        "draft_uniform_mix": draft_uniform_mix,
        "bootstrap_kind": bootstrap_kind,
        "random_draw_order": [
            "all_block_event_clock_uniforms",
            "draft_position_uniform_per_proposal",
            "draft_token_categorical_per_proposal",
            "verification_uniform_per_proposal",
            "positive_residual_categorical_after_rejection",
        ],
        "requests": requests,
        "aggregate": _aggregate_request_metrics(requests),
    }
    return (result, trace) if return_trace else result
