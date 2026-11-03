"""Released LLaDA2.1 integration for the exact S2D2 proposal.

This file does not modify or monkeypatch the official S2D2 source. The exact
sampler defined here has a different target from standalone S2D2. Standalone
S2D2 uses greedy block refinement, confidence thresholds, editing, and its 2L
routing policy. This adapter instead targets a natural event process: choose one
currently masked position uniformly and sample a non-mask token from that
position's temperature-one current-state row.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from .s2d2 import (
    draft_from_one_forward,
    full_probability_record,
    maximal_coupling_step,
    natural_event_target,
    sample_event,
)


def _uniform(generator: torch.Generator, device: torch.device) -> float:
    return float(torch.rand((), generator=generator, device=device).item())


def _make_generator(device: torch.device, seed: Optional[int]) -> torch.Generator:
    generator = torch.Generator(device=device)
    if seed is None:
        seed = int(torch.seed())
    generator.manual_seed(int(seed))
    return generator


def _prepare_state(model, input_ids, block_length: int, gen_length: int, mask_id: int):
    input_ids = input_ids.to(model.device)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("the minimal adapter supports batch size one")
    prompt_length = int(input_ids.shape[1])
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    state = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    state[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    cache = DynamicCache()
    nfe = 0
    if prefill_length:
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
        attention = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(model.dtype)
        )
        model(
            state[:, :prefill_length],
            attention_mask=attention[:, :, :prefill_length, :prefill_length],
            position_ids=position_ids[:, :prefill_length],
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )
        nfe = 1
    return state, position_ids, cache, prompt_length, num_blocks, prefill_blocks, nfe


def _finish_output(state, input_ids, prompt_length: int, gen_length: int, eos_id: int):
    generated = state[:, : prompt_length + gen_length]
    eos = (generated[0, prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
    stop = int(eos[0].item()) + 1 if len(eos) else gen_length
    return generated[:, : prompt_length + stop]


@torch.no_grad()
def generate_direct_natural_event(
    model,
    input_ids,
    attention_mask=None,
    block_length: int = 32,
    gen_length: int = 256,
    max_gen_toks: Optional[int] = None,
    eos_early_stop: bool = True,
    eos_id: int = 156892,
    mask_id: int = 156895,
    seed: Optional[int] = None,
    return_stats: bool = True,
    **kwargs,
):
    """Event-by-event exact reference for the defined natural event target."""
    del attention_mask, kwargs
    if max_gen_toks is not None:
        gen_length = int(max_gen_toks)
    state, positions, cache, prompt_length, num_blocks, first_block, nfe = _prepare_state(
        model, input_ids, block_length, gen_length, mask_id
    )
    generator = _make_generator(model.device, seed)
    event_records = []
    for block_number in range(first_block, num_blocks):
        start = block_number * block_length
        end = (block_number + 1) * block_length
        block = state[:, start:end].clone()
        block_positions = positions[:, start:end]
        while bool((block == mask_id).any()):
            logits = model(
                block,
                attention_mask=None,
                position_ids=block_positions,
                past_key_values=cache,
                use_cache=True,
                store_kv=False,
            ).logits[0]
            nfe += 1
            active = block[0] == mask_id
            target = natural_event_target(
                logits, active, temperature=1.0, forbidden_token_ids=(mask_id,)
            )
            event = sample_event(target, _uniform(generator, model.device))
            block[0, event[0]] = event[1]
            event_records.append(
                {
                    "position": start + event[0],
                    "token": event[1],
                    "target_probability": target.mass(event),
                }
            )
        state[:, start:end] = block
        if block_number < num_blocks - 1:
            model(
                block,
                attention_mask=None,
                position_ids=block_positions,
                past_key_values=cache,
                use_cache=True,
                store_kv=True,
            )
            nfe += 1
        if eos_early_stop and bool((state[0, prompt_length:end] == eos_id).any()):
            break
    output = _finish_output(state, input_ids, prompt_length, gen_length, eos_id)
    if not return_stats:
        return output
    return output, {
        "nfe": nfe,
        "target_calls": len(event_records),
        "proposal_calls": 0,
        "events": event_records,
        "target": "uniform masked position and temperature-one normalized non-mask token row",
    }


@torch.no_grad()
def generate_exact_s2d2_proposal(
    model,
    input_ids,
    attention_mask=None,
    block_length: int = 32,
    gen_length: int = 256,
    max_gen_toks: Optional[int] = None,
    eos_early_stop: bool = True,
    eos_id: int = 156892,
    mask_id: int = 156895,
    depth: int = 4,
    seed: Optional[int] = None,
    record_full_proposal: bool = False,
    return_stats: bool = True,
    **kwargs,
):
    """Use one current-state block forward to draft, then correct exactly."""
    del attention_mask, kwargs
    if max_gen_toks is not None:
        gen_length = int(max_gen_toks)
    if depth <= 0:
        raise ValueError("depth must be positive")
    state, positions, cache, prompt_length, num_blocks, first_block, nfe = _prepare_state(
        model, input_ids, block_length, gen_length, mask_id
    )
    generator = _make_generator(model.device, seed)
    proposal_calls = 0
    target_calls = 0
    accepted = 0
    corrected = 0
    event_records = []

    for block_number in range(first_block, num_blocks):
        start = block_number * block_length
        end = (block_number + 1) * block_length
        block = state[:, start:end].clone()
        block_positions = positions[:, start:end]
        while bool((block == mask_id).any()):
            active_at_round_start = block[0] == mask_id
            proposal_logits = model(
                block,
                attention_mask=None,
                position_ids=block_positions,
                past_key_values=cache,
                use_cache=True,
                store_kv=False,
            ).logits[0]
            nfe += 1
            proposal_calls += 1
            proposal_uniforms = [
                _uniform(generator, model.device)
                for _ in range(min(depth, int(active_at_round_start.sum().item())))
            ]
            draft = draft_from_one_forward(
                proposal_logits,
                active_at_round_start,
                depth=depth,
                uniforms=proposal_uniforms,
                token_temperature=1.0,
                forbidden_token_ids=(mask_id,),
                position_mode="confidence",
            )

            for draft_index, proposal in enumerate(draft):
                current_active = block[0] == mask_id
                if draft_index == 0:
                    target_logits = proposal_logits
                else:
                    target_logits = model(
                        block,
                        attention_mask=None,
                        position_ids=block_positions,
                        past_key_values=cache,
                        use_cache=True,
                        store_kv=False,
                    ).logits[0]
                    nfe += 1
                target_calls += 1
                target = natural_event_target(
                    target_logits,
                    current_active,
                    temperature=1.0,
                    forbidden_token_ids=(mask_id,),
                )
                result = maximal_coupling_step(
                    proposal,
                    target,
                    acceptance_uniform=_uniform(generator, model.device),
                    residual_uniform=_uniform(generator, model.device),
                )
                block[0, result.event[0]] = result.event[1]
                accepted += int(result.accepted)
                corrected += int(not result.accepted)
                record = {
                    "position": start + result.event[0],
                    "token": result.event[1],
                    "proposed_position": start + result.proposed_event[0],
                    "proposed_token": result.proposed_event[1],
                    "accepted": result.accepted,
                    "acceptance_probability": result.acceptance_probability,
                    "proposal_probability": result.proposal_probability,
                    "target_probability": result.target_probability,
                    "residual_mass": result.residual_mass,
                }
                if record_full_proposal:
                    record["full_proposal"] = full_probability_record(proposal)
                event_records.append(record)
                if not result.accepted:
                    break

        state[:, start:end] = block
        if block_number < num_blocks - 1:
            model(
                block,
                attention_mask=None,
                position_ids=block_positions,
                past_key_values=cache,
                use_cache=True,
                store_kv=True,
            )
            nfe += 1
        if eos_early_stop and bool((state[0, prompt_length:end] == eos_id).any()):
            break

    output = _finish_output(state, input_ids, prompt_length, gen_length, eos_id)
    if not return_stats:
        return output
    return output, {
        "nfe": nfe,
        "proposal_calls": proposal_calls,
        "target_calls": target_calls,
        "accepted_proposals": accepted,
        "positive_residual_corrections": corrected,
        "events": event_records,
        "proposal": "confidence-normalized position and frozen temperature-one non-mask token rows",
        "target": "uniform masked position and current temperature-one normalized non-mask token row",
        "depth": depth,
    }
