from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import torch

from specctmc import ctmc as runtime
from specctmc.cache import PromptOnlyTargetCacheModel

MASK_ID = 4
VOCAB_SIZE = 5


class FakeCacheCapableLLaDA(torch.nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, width=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, width)
        self.head = torch.nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(
            weight_tying=False,
            scale_logits=False,
            d_model=width,
            use_cache=False,
        )
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.1, 0.4, 0.3],
                        [-0.3, 0.5, 0.1, -0.2],
                        [0.6, 0.2, -0.4, 0.1],
                        [0.1, -0.5, 0.3, 0.7],
                        [-0.2, 0.1, 0.2, -0.1],
                    ],
                    dtype=torch.float32,
                )
            )
            self.head.weight.copy_(
                torch.tensor(
                    [
                        [0.7, -0.2, 0.1, 0.4],
                        [-0.1, 0.8, -0.3, 0.2],
                        [0.2, 0.1, 0.9, -0.4],
                        [-0.5, 0.3, 0.2, 0.8],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
        self.full_calls = 0
        self.cached_calls = 0

    @property
    def device(self):
        return next(self.parameters()).device

    def get_output_embeddings(self):
        return self.head

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=False,
        num_logits_to_keep=0,
        **kwargs,
    ):
        embedded = self.embedding(input_ids)
        if past_key_values is None:
            self.full_calls += 1
            global_context = embedded.mean(dim=1, keepdim=True)
            keys = embedded + 0.2 * global_context
            values = embedded - 0.1 * global_context
            complete_keys = keys
            complete_values = values
        else:
            self.cached_calls += 1
            past_keys, past_values = past_key_values[0]
            suffix_context = embedded.mean(dim=1, keepdim=True)
            keys = embedded + 0.2 * suffix_context
            values = embedded - 0.1 * suffix_context
            complete_keys = torch.cat((past_keys[:, 0], keys), dim=1)
            complete_values = torch.cat((past_values[:, 0], values), dim=1)
        context = complete_keys.mean(dim=1, keepdim=True) + 0.3 * complete_values.mean(dim=1, keepdim=True)
        hidden = embedded + context
        logits = self.head(hidden)
        if num_logits_to_keep:
            logits = logits[:, -num_logits_to_keep:, :]
        returned_cache = None
        if use_cache:
            returned_cache = ((complete_keys.unsqueeze(1), complete_values.unsqueeze(1)),)
        return SimpleNamespace(logits=logits, past_key_values=returned_cache)


def make_states(batch=4, suffix=32):
    row = [0, 1] + [MASK_ID] * suffix
    return torch.tensor([row for _ in range(batch)], dtype=torch.long)


def wrapper_state_machine_checks():
    inner = FakeCacheCapableLLaDA()
    wrapper = PromptOnlyTargetCacheModel(inner, refresh_interval=8)
    wrapper.begin_request(prompt_length=2, target_batch_size=4)
    states = make_states()
    first = wrapper(states, num_logits_to_keep=32).logits
    assert first.shape == (4, 32, VOCAB_SIZE)
    assert wrapper.metrics()["prompt_cache_refreshes"] == 1
    assert wrapper.remaining_before_refresh() == 8
    wrapper.note_commits(7)
    assert wrapper.remaining_before_refresh() == 1
    wrapper(states, num_logits_to_keep=32)
    assert wrapper.metrics()["prompt_cache_refreshes"] == 1
    wrapper.note_commits(1)
    assert wrapper.remaining_before_refresh() == 8
    changed = states.clone()
    changed[:, 2] = 3
    wrapper(changed, num_logits_to_keep=32)
    assert wrapper.metrics()["prompt_cache_refreshes"] == 2
    trace = wrapper.trace()
    refresh_totals = [event["total_committed_reveals"] for event in trace["events"] if event["kind"] == "prompt_cache_refresh"]
    assert refresh_totals == [0, 8]
    for layer in wrapper._prompt_past_key_values:
        assert layer[0].shape[-2] == 2
        assert layer[1].shape[-2] == 2
    return {
        "refresh_totals": refresh_totals,
        "prompt_cache_length": 2,
        "suffix_recomputed_length": 32,
    }


def target_probabilities(model, states, prompt_length=2):
    prompt_masks = torch.zeros_like(states, dtype=torch.bool)
    prompt_masks[:, :prompt_length] = True
    active_masks = ~prompt_masks
    metrics = runtime._new_metrics()
    probabilities, _, _ = runtime._target_probabilities(
        model=model,
        states=states,
        times=torch.full((states.shape[0],), 0.5),
        prompt_masks=prompt_masks,
        active_masks=active_masks,
        metrics=metrics,
        mask_id=MASK_ID,
        temperature=1.0,
        cfg_scale=0.0,
        target_evaluation_batch_size=4,
    )
    return probabilities


def cached_process_is_shared_and_differs_from_full_forward():
    initial = make_states()
    changed = initial.clone()
    changed[:, 2] = 3
    sequential_inner = FakeCacheCapableLLaDA()
    speculative_inner = copy.deepcopy(sequential_inner)
    sequential = PromptOnlyTargetCacheModel(sequential_inner, 8)
    speculative = PromptOnlyTargetCacheModel(speculative_inner, 8)
    sequential.begin_request(2, 4)
    speculative.begin_request(2, 4)
    target_probabilities(sequential, initial)
    target_probabilities(speculative, initial)
    sequential_cached = target_probabilities(sequential, changed)
    speculative_cached = target_probabilities(speculative, changed)
    torch.testing.assert_close(sequential_cached, speculative_cached, rtol=0, atol=0)
    full = target_probabilities(copy.deepcopy(sequential_inner), changed)
    maximum_difference = float((sequential_cached - full).abs().max().item())
    assert maximum_difference > 0
    assert sequential.trace()["persistent_cache_updates_from_tentative_states"] == 0
    return {
        "cached_sequential_and_speculative_target_equal": True,
        "cached_vs_full_forward_maximum_probability_difference": maximum_difference,
        "persistent_tentative_cache_updates": 0,
    }


def run_method(method, interval):
    inner = FakeCacheCapableLLaDA()
    model = PromptOnlyTargetCacheModel(inner, interval)
    prompt = torch.tensor([[0, 1]], dtype=torch.long)
    event_times = [0.99 - 0.025 * index for index in range(32)]
    common = dict(
        model=model,
        prompt=prompt,
        batch_size=1,
        gen_length=32,
        block_length=32,
        mask_id=MASK_ID,
        vocab_size=VOCAB_SIZE,
        cfg_scale=0.0,
        temperature=1.0,
        eps=0.01,
        schedule="loglinear",
        seed=8123,
        event_times_by_block=[event_times],
        target_evaluation_batch_size=4,
        return_trace=True,
    )
    if method == "sequential":
        return runtime.generate_ctmc_exact(
            stream_seeds={"event": 100, "target": 200},
            **common,
        )
    return runtime.generate_ctmc_speculative(
        speculative_depth=6,
        draft_kind="stale_logits",
        draft_uniform_mix=0.0001,
        bootstrap_kind="uniform",
        stream_seeds={"event": 100, "draft": 300, "verification": 400, "residual": 500},
        selected_position_projection=False,
        verification_mixture_lambda=0.0,
        **common,
    )


def cached_next_event_reconstruction_checks():
    report = {}
    for interval in (8, 16, 32):
        inner = FakeCacheCapableLLaDA()
        model = PromptOnlyTargetCacheModel(inner, interval)
        initial = make_states(batch=4)
        current = initial.clone()
        current[:, 2] = 3
        model.begin_request(prompt_length=2, target_batch_size=4)
        stale = target_probabilities(model, initial)[0]
        current_target = target_probabilities(model, current)[0]
        mutable = current[0, 2:] == MASK_ID
        draft = stale.clone()
        draft[~mutable] = 0.0
        draft = draft / draft.sum()
        target = current_target / current_target.sum()
        accepted_mass = torch.minimum(target, draft)
        rejection_mass = 1.0 - accepted_mass.sum()
        positive_residual = torch.clamp(target - draft, min=0.0)
        assert torch.allclose(rejection_mass, positive_residual.sum(), rtol=1e-6, atol=1e-7)
        residual = positive_residual / rejection_mass
        reconstructed = accepted_mass + rejection_mass * residual
        maximum_error = float((reconstructed - target).abs().max().item())
        assert maximum_error <= 1e-7
        metrics = model.metrics()
        assert metrics["prompt_cache_refreshes"] == 1
        assert metrics["prompt_cache_construction_seconds"] >= 0.0
        assert metrics["prompt_cache_periodic_refresh_seconds"] == 0.0
        assert metrics["prompt_cache_total_refresh_seconds"] == metrics["prompt_cache_construction_seconds"]
        assert metrics["prompt_cache_copy_seconds"] <= metrics["prompt_cache_total_refresh_seconds"] + 1e-12
        assert metrics["prompt_cache_suffix_forward_seconds"] >= 0.0
        report[str(interval)] = {
            "maximum_next_event_reconstruction_error": maximum_error,
            "rejection_mass": float(rejection_mass.item()),
            "target_joint_mass": float(target.sum().item()),
            "draft_joint_mass": float(draft.sum().item()),
            "cache_state_digest": model.trace()["events"][0]["state_sha256"],
        }
    return report


def runtime_refresh_boundary_checks():
    report = {}
    for interval in (8, 16, 32):
        sequential_output, sequential_trace = run_method("sequential", interval)
        speculative_output, speculative_trace = run_method("speculative", interval)
        assert sequential_output.shape == speculative_output.shape == (1, 34)
        assert sequential_trace["preserved_sequential_distribution"] == "prompt_cached_sequential_ctmc"
        assert speculative_trace["preserved_sequential_distribution"] == "prompt_cached_sequential_ctmc"
        assert sequential_trace["requests"][0]["event_times"] == speculative_trace["requests"][0]["event_times"]
        expected_refreshes = 32 // interval
        assert sequential_trace["aggregate"]["prompt_cache_refreshes"] == expected_refreshes
        assert speculative_trace["aggregate"]["prompt_cache_refreshes"] == expected_refreshes
        capped_rounds = 0
        for round_record in speculative_trace["requests"][0]["rounds"]:
            if round_record.get("kind") == "terminal_cleanup":
                continue
            assert round_record["realized_depth"] <= round_record["cache_remaining_before_round"]
            assert round_record["effective_depth_before_cache_boundary"] <= interval
            if round_record["effective_depth_before_cache_boundary"] < 6:
                capped_rounds += 1
        assert capped_rounds > 0 if interval in (8, 16) else True
        prompt_cache_trace = speculative_trace["requests"][0]["prompt_cache_trace"]
        refresh_totals = [
            event["total_committed_reveals"]
            for event in prompt_cache_trace["events"]
            if event["kind"] == "prompt_cache_refresh"
        ]
        assert refresh_totals == list(range(0, 32, interval))
        report[str(interval)] = {
            "sequential_refreshes": sequential_trace["aggregate"]["prompt_cache_refreshes"],
            "speculative_refreshes": speculative_trace["aggregate"]["prompt_cache_refreshes"],
            "refresh_totals": refresh_totals,
            "speculative_boundary_capped_rounds": capped_rounds,
            "event_clock_equal": True,
            "sequential_output_tokens": sequential_output[0, 2:].tolist(),
            "speculative_output_tokens": speculative_output[0, 2:].tolist(),
        }
    return report


def invalid_composition_check():
    model = PromptOnlyTargetCacheModel(FakeCacheCapableLLaDA(), 8)
    prompt = torch.tensor([[0, 1]], dtype=torch.long)
    try:
        runtime.generate_ctmc_speculative(
            model=model,
            prompt=prompt,
            gen_length=8,
            block_length=8,
            speculative_depth=3,
            target_evaluation_batch_size=4,
            mask_id=MASK_ID,
            vocab_size=VOCAB_SIZE,
            selected_position_projection=True,
        )
    except ValueError as error:
        assert "separate composition test" in str(error)
    else:
        raise AssertionError("cache plus selected projection was not gated")
    return True


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_wrapper_state_machine(self):
        wrapper_state_machine_checks()

    def test_cached_process_matches_between_samplers(self):
        cached_process_is_shared_and_differs_from_full_forward()

    def test_next_event_reconstruction(self):
        cached_next_event_reconstruction_checks()

    def test_runtime_refresh_boundaries(self):
        runtime_refresh_boundary_checks()

    def test_unsupported_cache_projection_combination_is_rejected(self):
        self.assertTrue(invalid_composition_check())


if __name__ == "__main__":
    unittest.main(verbosity=2)
