"""Exact fixed-batch CTMC simulation for Dream masked language models.

Dream and LLaDA share the absorbing-mask event clock after the mask token is
excluded from the clean-token distribution. Dream differs in the model output
used for each position. Its official generation code shifts logits one position
to the right before selecting masked positions. This module applies that shift
exactly once and delegates the event loop to the validated fixed-batch sampler.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from . import ctmc_fixed as fixed_batch

DREAM_MASK_ID = 151666
DREAM_VOCAB_SIZE = 152064


def shift_dream_logits(logits: torch.Tensor) -> torch.Tensor:
    """Apply the position shift in Dream's official generation path."""
    if logits.ndim != 3 or logits.shape[1] == 0:
        raise ValueError("Dream logits must have shape [batch, sequence_length, vocab]")
    return torch.cat((logits[:, :1], logits[:, :-1]), dim=1)


class DreamShiftedLogitsModel(torch.nn.Module):
    """Expose Dream's shifted generation logits through the LLaDA model contract."""

    def __init__(self, model: Any):
        super().__init__()
        self.inner_model = model
        self.target_model_family = "dream"

    @property
    def device(self) -> torch.device:
        device = getattr(self.inner_model, "device", None)
        if device is not None:
            return torch.device(device)
        return next(self.inner_model.parameters()).device

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        output = self.inner_model(input_ids, "full", None)
        logits = output.logits
        if logits.shape[:2] != input_ids.shape:
            raise RuntimeError("Dream model logits do not align with input identifiers")
        if logits.shape[-1] != DREAM_VOCAB_SIZE:
            raise RuntimeError(
                f"Dream model vocabulary width must equal {DREAM_VOCAB_SIZE}, "
                f"received {logits.shape[-1]}"
            )
        return SimpleNamespace(logits=shift_dream_logits(logits))


def wrap_dream_model(model: Any) -> DreamShiftedLogitsModel:
    if isinstance(model, DreamShiftedLogitsModel):
        return model
    return DreamShiftedLogitsModel(model)


@torch.no_grad()
def dream_token_probabilities(
    model: Any,
    x: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
    mask_id: int = DREAM_MASK_ID,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
) -> torch.Tensor:
    if mask_id != DREAM_MASK_ID:
        raise ValueError(f"Dream mask_id must equal {DREAM_MASK_ID}")
    if cfg_scale != 0.0:
        raise ValueError("Dream exact evaluation currently requires cfg_scale equal to zero")
    return fixed_batch.llada_token_probabilities(
        model=wrap_dream_model(model),
        x=x,
        prompt_mask=prompt_mask,
        active_mask=active_mask,
        mask_id=mask_id,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )


@torch.no_grad()
def dream_qrate(
    model: Any,
    x: torch.Tensor,
    t: torch.Tensor,
    prompt_mask: torch.Tensor,
    active_mask: torch.Tensor,
    mask_id: int = DREAM_MASK_ID,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
) -> torch.Tensor:
    if mask_id != DREAM_MASK_ID:
        raise ValueError(f"Dream mask_id must equal {DREAM_MASK_ID}")
    if cfg_scale != 0.0:
        raise ValueError("Dream exact evaluation currently requires cfg_scale equal to zero")
    return fixed_batch.llada_qrate(
        model=wrap_dream_model(model),
        x=x,
        t=t,
        prompt_mask=prompt_mask,
        active_mask=active_mask,
        mask_id=mask_id,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )


def _validate_generation_parameters(mask_id: int, vocab_size: int, cfg_scale: float) -> None:
    if mask_id != DREAM_MASK_ID:
        raise ValueError(f"Dream mask_id must equal {DREAM_MASK_ID}")
    if vocab_size != DREAM_VOCAB_SIZE:
        raise ValueError(f"Dream vocab_size must equal {DREAM_VOCAB_SIZE}")
    if cfg_scale != 0.0:
        raise ValueError("Dream exact evaluation currently requires cfg_scale equal to zero")


def generate_ctmc_exact(model: Any, **kwargs):
    _validate_generation_parameters(
        int(kwargs.get("mask_id", DREAM_MASK_ID)),
        int(kwargs.get("vocab_size", DREAM_VOCAB_SIZE)),
        float(kwargs.get("cfg_scale", 0.0)),
    )
    kwargs["mask_id"] = DREAM_MASK_ID
    kwargs["vocab_size"] = DREAM_VOCAB_SIZE
    kwargs["cfg_scale"] = 0.0
    return fixed_batch.generate_ctmc_exact(model=wrap_dream_model(model), **kwargs)


def generate_ctmc_speculative(model: Any, **kwargs):
    _validate_generation_parameters(
        int(kwargs.get("mask_id", DREAM_MASK_ID)),
        int(kwargs.get("vocab_size", DREAM_VOCAB_SIZE)),
        float(kwargs.get("cfg_scale", 0.0)),
    )
    kwargs["mask_id"] = DREAM_MASK_ID
    kwargs["vocab_size"] = DREAM_VOCAB_SIZE
    kwargs["cfg_scale"] = 0.0
    return fixed_batch.generate_ctmc_speculative(model=wrap_dream_model(model), **kwargs)


precompute_event_times = fixed_batch.precompute_event_times
