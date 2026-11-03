"""Selective output-head projection for exact speculative CTMC verification.

The transformer runs with the existing fixed input batch shape. Vocabulary logits are
computed only for proposed positions, then for mutable positions of the reached
candidate prestate. Dream alignment is applied to hidden-state positions before the
output head, so this path never shifts a complete vocabulary-logit tensor.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F


class _StopAfterFinalNorm(BaseException):
    """Internal control-flow signal used to bypass LLaDA's output head."""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_dream(model: Any) -> bool:
    return getattr(model, "target_model_family", None) == "dream"


def _llada_base(model: Any) -> Any:
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "transformer"):
        raise TypeError("selected LLaDA projection requires model.model.transformer")
    return base


@torch.no_grad()
def aligned_suffix_hidden_states(
    model: Any,
    states: torch.Tensor,
    prompt_length: int,
) -> Tuple[torch.Tensor, float, float]:
    """Return final normalized hidden states aligned to generated output positions.

    Returns ``(hidden, transformer_seconds, alignment_seconds)``. Hidden has shape
    ``[batch, suffix_length, hidden_width]``.
    """
    if states.ndim != 2:
        raise ValueError("states must have shape [batch, sequence_length]")
    if not 0 <= prompt_length < states.shape[1]:
        raise ValueError("prompt_length must identify a nonempty generated suffix")
    device = states.device

    if _is_dream(model):
        inner = getattr(model, "inner_model", None)
        if inner is None or not hasattr(inner, "model"):
            raise TypeError("selected Dream projection requires the wrapped inner model")
        _sync(device)
        started = time.perf_counter()
        outputs = inner.model(
            input_ids=states,
            attention_mask="full",
            position_ids=None,
            return_dict=True,
        )
        _sync(device)
        transformer_seconds = time.perf_counter() - started
        final_hidden = outputs[0]
        if final_hidden.shape[:2] != states.shape:
            raise RuntimeError("Dream base model returned misaligned hidden states")
        _sync(device)
        alignment_started = time.perf_counter()
        output_positions = torch.arange(
            prompt_length, states.shape[1], device=device, dtype=torch.long
        )
        hidden_positions = torch.clamp(output_positions - 1, min=0)
        aligned = final_hidden.index_select(1, hidden_positions)
        _sync(device)
        alignment_seconds = time.perf_counter() - alignment_started
        return aligned, transformer_seconds, alignment_seconds

    base = _llada_base(model)
    final_norm = getattr(base.transformer, "ln_f", None)
    if final_norm is None:
        raise TypeError("selected LLaDA projection requires the final layer norm")
    captured = []

    def stop_after_norm(_module, _inputs, output):
        captured.append(output)
        raise _StopAfterFinalNorm()

    handle = final_norm.register_forward_hook(stop_after_norm)
    _sync(device)
    started = time.perf_counter()
    try:
        model(states, num_logits_to_keep=states.shape[1] - prompt_length)
    except _StopAfterFinalNorm:
        pass
    finally:
        handle.remove()
    _sync(device)
    transformer_seconds = time.perf_counter() - started
    if len(captured) != 1:
        raise RuntimeError("LLaDA final hidden state was not captured exactly once")
    final_hidden = captured[0]
    if final_hidden.shape[:2] != states.shape:
        raise RuntimeError("LLaDA model returned misaligned hidden states")
    return final_hidden[:, prompt_length:, :], transformer_seconds, 0.0


def _output_head_and_config(model: Any) -> Tuple[Any, Any, str]:
    if _is_dream(model):
        owner = model.inner_model
        family = "dream"
    else:
        owner = model
        family = "llada"
    head = owner.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise TypeError("model output head is unavailable")
    config = getattr(owner, "config", None)
    if config is None and family == "llada":
        config = getattr(_llada_base(owner), "config", None)
    if config is None:
        raise TypeError("model configuration is unavailable")
    return head, config, family


@torch.no_grad()
def project_hidden_logits(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the released model's vocabulary head to selected hidden vectors."""
    head, config, family = _output_head_and_config(model)
    if family == "llada" and bool(getattr(config, "weight_tying", False)):
        logits = F.linear(hidden, head.weight, None)
    else:
        logits = head(hidden)
    if family == "llada" and bool(getattr(config, "scale_logits", False)):
        logits.mul_(1.0 / math.sqrt(float(config.d_model)))
    return logits


@torch.no_grad()
def probabilities_for_hidden(
    model: Any,
    hidden: torch.Tensor,
    mask_id: int,
    temperature: float,
) -> Tuple[torch.Tensor, float]:
    """Project selected hidden vectors and normalize over clean tokens."""
    if temperature <= 0:
        raise ValueError("temperature must be strictly positive")
    device = hidden.device
    _sync(device)
    started = time.perf_counter()
    logits = project_hidden_logits(model, hidden).float()
    if not 0 <= mask_id < logits.shape[-1]:
        raise ValueError("mask_id is outside the model vocabulary")
    logits[..., mask_id] = -float("inf")
    probabilities = torch.softmax(logits / float(temperature), dim=-1)
    _sync(device)
    elapsed = time.perf_counter() - started
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=2e-5, rtol=2e-5):
        raise RuntimeError("selected target probability rows do not sum to one")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("selected target probabilities are invalid")
    return probabilities, elapsed


@torch.no_grad()
def complete_reached_probability_matrix(
    model: Any,
    aligned_hidden: torch.Tensor,
    mutable: torch.Tensor,
    mask_id: int,
    temperature: float,
    selected_position: Optional[int] = None,
    selected_probability: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, int]:
    """Project mutable rows for one reached prestate, reusing its selected row."""
    if aligned_hidden.ndim != 2:
        raise ValueError("aligned_hidden must have shape [suffix_length, hidden_width]")
    if mutable.shape != aligned_hidden.shape[:1] or mutable.dtype != torch.bool:
        raise ValueError("mutable must be a Boolean suffix-position mask")
    if (selected_position is None) != (selected_probability is None):
        raise ValueError("selected position and probability must be supplied together")
    positions = torch.nonzero(mutable, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("reached prestate has no mutable position")
    if selected_position is not None:
        if not 0 <= selected_position < mutable.numel() or not bool(mutable[selected_position].item()):
            raise ValueError("selected position is not mutable in the reached prestate")
        positions_to_project = positions[positions != selected_position]
    else:
        positions_to_project = positions
    head, _config, _family = _output_head_and_config(model)
    vocab_size = int(head.weight.shape[0])
    output = torch.zeros(
        (aligned_hidden.shape[0], vocab_size),
        device=aligned_hidden.device,
        dtype=torch.float32,
    )
    elapsed = 0.0
    if positions_to_project.numel():
        probabilities, elapsed = probabilities_for_hidden(
            model=model,
            hidden=aligned_hidden.index_select(0, positions_to_project),
            mask_id=mask_id,
            temperature=temperature,
        )
        output.index_copy_(0, positions_to_project, probabilities)
    if selected_position is not None:
        if selected_probability.ndim != 1 or selected_probability.shape[0] != vocab_size:
            raise ValueError("selected probability has the wrong vocabulary width")
        output[selected_position].copy_(selected_probability)
    row_sums = output.sum(dim=-1)
    if not torch.allclose(
        row_sums[mutable], torch.ones_like(row_sums[mutable]), atol=2e-5, rtol=2e-5
    ):
        raise RuntimeError("reached-state active probability rows do not sum to one")
    if torch.count_nonzero(row_sums[~mutable]):
        raise RuntimeError("reached-state inactive probability rows are not zero")
    return output, elapsed, int(positions_to_project.numel())
