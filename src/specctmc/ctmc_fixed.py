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
    model: Any,
    x: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
    mask_id: int = MASK_ID,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
) -> torch.Tensor:
    """Return target token probabilities with inactive rows set to zero."""
    _validate_masks(x, prompt_mask, active_mask)
    if temperature <= 0:
        raise ValueError("temperature must be strictly positive for CTMC sampling")
    if cfg_scale < 0:
        raise ValueError("cfg_scale must be nonnegative")

    if cfg_scale > 0:
        unconditional_x = x.clone()
        unconditional_x[prompt_mask] = mask_id
        model_x = torch.cat([x, unconditional_x], dim=0)
        logits = model(model_x).logits
        conditional_logits, unconditional_logits = torch.chunk(logits, 2, dim=0)
        logits = unconditional_logits + (cfg_scale + 1.0) * (
            conditional_logits - unconditional_logits
        )
    else:
        logits = model(x).logits

    if logits.ndim != 3 or logits.shape[:2] != x.shape:
        raise RuntimeError("model logits must have shape [batch, sequence_length, vocab]")
    if not 0 <= mask_id < logits.shape[-1]:
        raise ValueError("mask_id is outside the model vocabulary")

    logits = logits.float()
    logits[:, :, mask_id] = -float("inf")
    probabilities = torch.softmax(logits / float(temperature), dim=-1)
    mutable = active_mask & (~prompt_mask) & (x == mask_id)
    probabilities = probabilities * mutable.unsqueeze(-1)
    probabilities.scatter_(-1, x.unsqueeze(-1), 0.0)

    if not torch.isfinite(probabilities).all():
        raise RuntimeError("target probabilities contain non-finite values")
    if (probabilities < 0).any():
        raise RuntimeError("target probabilities contain negative values")

    active_sums = probabilities.sum(dim=-1)[mutable]
    if active_sums.numel() and not torch.allclose(
        active_sums,
        torch.ones_like(active_sums),
        atol=2e-5,
        rtol=2e-5,
    ):
        maximum_error = float((active_sums - 1.0).abs().max().item())
        raise RuntimeError(
            f"active target probability rows do not sum to one, max error {maximum_error}"
        )
    inactive = ~mutable
    if probabilities[inactive].numel() and torch.count_nonzero(
        probabilities[inactive]
    ):
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

    mutable = active_mask & (~prompt_mask) & (x == mask_id)
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
        "draft_seconds": 0.0,
        "residual_seconds": 0.0,
        "bootstrap_target_rows": 0,
        "terminal_cleanup_rows": 0,
        "natural_ctmc_jumps": 0,
        "residual_corrections": 0,
        "accepted_draft_marks": 0,
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
        empty = torch.empty(
            (0, states.shape[1], 0), device=states.device, dtype=torch.float32
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
    x[remaining] = probabilities.argmax(dim=-1)[remaining]
    metrics["terminal_cleanup_rows"] += 1
    return count


def _aggregate_request_metrics(requests: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "sequential_target_rounds",
        "total_target_rate_rows",
        "total_model_input_rows",
        "target_padding_rows",
        "target_forward_seconds",
        "draft_seconds",
        "residual_seconds",
        "bootstrap_target_rows",
        "terminal_cleanup_rows",
        "natural_ctmc_jumps",
        "residual_corrections",
        "accepted_draft_marks",
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
            position, token = _sample_mark(probabilities[0], target_generator)
            if not bool(mutable[0, position].item()) or token == mask_id:
                raise RuntimeError("a natural CTMC jump selected an invalid mark")
            x[0, position] = token
            if int((active_mask & (x == mask_id)).sum().item()) != mask_count - 1:
                raise RuntimeError("a CTMC jump did not reduce the mask count by one")
            metrics["natural_ctmc_jumps"] += 1
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

    request = {
        "seed": int(seed),
        "stream_seeds": seeds,
        "precomputed_event_times": event_times_by_block is not None,
        "method": "ctmc_exact",
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
    x: torch.Tensor,
    active_mask: torch.Tensor,
    stale_cache: Optional[torch.Tensor],
    vocab_size: int,
    mask_id: int,
    uniform_mix: float,
    exact_cache_for_state: bool,
) -> torch.Tensor:
    if not 0 <= uniform_mix < 1:
        raise ValueError("draft_uniform_mix must lie in [0, 1)")
    mutable = active_mask[0] & (x[0] == mask_id)
    positions = torch.nonzero(mutable, as_tuple=False).flatten()
    output = torch.zeros(
        (x.shape[1], vocab_size), device=x.device, dtype=torch.float32
    )
    if positions.numel() == 0:
        return output
    uniform = _uniform_nonmask(
        int(positions.numel()), vocab_size, mask_id, x.device
    )
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
    if not torch.allclose(
        active_sums,
        torch.ones_like(active_sums),
        atol=2e-5,
        rtol=2e-5,
    ):
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
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if speculative_depth <= 0:
        raise ValueError("speculative_depth must be positive")
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
            round_event_times = block_event_times[
                event_cursor : event_cursor + speculative_depth
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
                    stale_cache=None if draft_kind == "uniform" else stale_cache,
                    vocab_size=vocab_size,
                    mask_id=mask_id,
                    uniform_mix=0.0 if draft_kind == "uniform" else draft_uniform_mix,
                    exact_cache_for_state=False if draft_kind == "uniform" else exact_cache,
                )
                position, token = _sample_mark(draft_probability, draft_generator)
                if not bool(draft_mutable[0, position].item()) or token == mask_id:
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
                draft_state[0, position] = token

            draft_seconds = time.perf_counter() - draft_started
            metrics["draft_seconds"] += draft_seconds
            realized_depth = len(draft_prestates)
            if realized_depth == 0:
                raise RuntimeError("event drafting produced no proposals")
            metrics["drafted_marks"] += realized_depth
            metrics["speculative_rounds"] += 1

            reuse_first = cache_is_exact_for_current_state
            target_start = 1 if reuse_first else 0
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
                target_probabilities = torch.empty(
                    (0, x.shape[1], vocab_size),
                    device=x.device,
                    dtype=torch.float32,
                )
                target_microbatches = 0
                target_seconds = 0.0

            accepted = 0
            rejected = False
            correction_position: Optional[int] = None
            correction_token: Optional[int] = None
            acceptance_probabilities: List[float] = []
            committed_times: List[float] = []
            latest_target_probability: Optional[torch.Tensor] = None

            for proposal_index in range(realized_depth):
                prestate = draft_prestates[proposal_index]
                if not torch.equal(x, prestate):
                    raise RuntimeError(
                        "a verified draft prestate does not match the committed state"
                    )
                position = draft_positions[proposal_index]
                token = draft_tokens[proposal_index]
                draft_probability = draft_matrices[proposal_index]
                if proposal_index < target_start:
                    if stale_cache is None:
                        raise RuntimeError("target bootstrap cache is absent")
                    target_probability = stale_cache
                else:
                    target_probability = target_probabilities[
                        proposal_index - target_start
                    ]

                recorded = draft_proposal_probabilities[proposal_index]
                used = float(draft_probability[position, token].item())
                if not math.isclose(recorded, used, rel_tol=0.0, abs_tol=1e-8):
                    raise RuntimeError(
                        "recorded draft probability changed before verification"
                    )
                target_value = float(target_probability[position, token].item())
                if used <= 0:
                    raise RuntimeError("draft proposal had zero probability")
                accept_probability = min(1.0, target_value / used)
                acceptance_probabilities.append(accept_probability)

                accept = (
                    proposal_index < target_start
                    or verification_uniforms[proposal_index] < accept_probability
                )
                latest_target_probability = target_probability
                cache_is_exact_for_current_state = False
                if accept:
                    x[0, position] = token
                    accepted += 1
                    metrics["accepted_draft_marks"] += 1
                    metrics["natural_ctmc_jumps"] += 1
                    committed_times.append(draft_times[proposal_index])
                    continue

                residual_mutable = active_mask[0] & (prestate[0] == mask_id)
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
                x[0, correction_position] = correction_token
                metrics["residual_corrections"] += 1
                metrics["natural_ctmc_jumps"] += 1
                committed_times.append(draft_times[proposal_index])
                rejected = True
                break

            if latest_target_probability is None:
                raise RuntimeError("verification produced no target probability")
            stale_cache = latest_target_probability.detach().clone()
            committed_count = accepted + int(rejected)
            end_mask_count = int((active_mask & (x == mask_id)).sum().item())
            if end_mask_count != start_mask_count - committed_count:
                raise RuntimeError(
                    "committed speculative marks did not reduce mask count correctly"
                )
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
                    "realized_depth": realized_depth,
                    "accepted_draft_depth": accepted,
                    "residual_correction": rejected,
                    "corrected_position": correction_position,
                    "corrected_token": correction_token,
                    "bootstrap_target_rows": bootstrap_rows_pending,
                    "reused_bootstrap_first_row": reuse_first,
                    "target_rate_rows": realized_depth - target_start,
                    "target_evaluation_calls": target_microbatches,
                    "target_forward_seconds": target_seconds,
                    "draft_seconds": draft_seconds,
                    "acceptance_probabilities": acceptance_probabilities,
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

    request = {
        "seed": int(seed),
        "stream_seeds": seeds,
        "precomputed_event_times": event_times_by_block is not None,
        "method": "ctmc_speculative_fixed_batch",
        "speculative_depth": speculative_depth,
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
        )
        request["request_index"] = request_index
        outputs.append(output)
        requests.append(request)
    result = torch.cat(outputs, dim=0)
    trace = {
        "schema_version": 1,
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
        "finite_precision_evaluator": "fixed_batch_duplicate_padding",
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
