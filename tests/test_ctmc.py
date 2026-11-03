from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from specctmc import approximate, ctmc


class FakeMaskedModel(torch.nn.Module):
    def __init__(self, vocabulary_size: int = 7):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen_batch_sizes: list[int] = []

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(self, state: torch.Tensor, **kwargs) -> SimpleNamespace:
        num_logits_to_keep = int(kwargs.pop("num_logits_to_keep", 0))
        batch_size, sequence_length = state.shape
        self.seen_batch_sizes.append(batch_size)
        tokens = torch.arange(
            self.vocabulary_size, dtype=torch.float32, device=state.device
        ).view(1, 1, -1)
        positions = torch.arange(
            sequence_length, dtype=torch.float32, device=state.device
        ).view(1, -1, 1)
        logits = tokens * 0.09 + positions * 0.01
        logits = logits + (state.float().unsqueeze(-1) % 3) * tokens * 0.003
        logits = logits.expand(batch_size, -1, -1).clone()
        if num_logits_to_keep:
            logits = logits[:, -num_logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def common(model: FakeMaskedModel) -> dict:
    return {
        "model": model,
        "prompt": torch.tensor([[1, 2]], dtype=torch.long),
        "batch_size": 1,
        "gen_length": 4,
        "block_length": 4,
        "mask_id": 6,
        "vocab_size": 7,
        "temperature": 1.0,
        "eps": 0.1,
        "seed": 11,
        "target_evaluation_batch_size": 4,
        "return_trace": True,
    }


class ExactAndApproximateSamplerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_direct_ctmc_completes_without_masks(self):
        model = FakeMaskedModel()
        output, trace = ctmc.generate_ctmc_exact(
            event_times_by_block=[[0.8, 0.6, 0.4]],
            stream_seeds={"event": 11, "target": 12},
            **common(model),
        )
        self.assertNotIn(6, output[0, 2:].tolist())
        self.assertGreater(trace["aggregate"]["sequential_target_rounds"], 0)
        self.assertEqual(set(model.seen_batch_sizes), {4})

    def test_spec_ctmc_completes_without_masks(self):
        model = FakeMaskedModel()
        output, trace = ctmc.generate_ctmc_speculative(
            speculative_depth=3,
            draft_kind="stale_logits",
            bootstrap_kind="uniform",
            event_times_by_block=[[0.8, 0.6, 0.4]],
            stream_seeds={
                "event": 11,
                "draft": 13,
                "verification": 14,
                "residual": 15,
            },
            **common(model),
        )
        self.assertNotIn(6, output[0, 2:].tolist())
        self.assertGreater(trace["aggregate"]["sequential_target_rounds"], 0)
        self.assertEqual(set(model.seen_batch_sizes), {4})

    def test_tau_leaping_uses_the_validated_rate_function(self):
        model = FakeMaskedModel()
        output, trace = approximate.generate_tau_leaping(
            rate_function=ctmc.llada_qrate,
            steps=4,
            **common(model),
        )
        self.assertNotIn(6, output[0, 2:].tolist())
        self.assertEqual(trace["method"], "approximate_tau_leaping")
        self.assertGreater(trace["metrics"]["sequential_target_rounds"], 0)

    def test_trapezoidal_solver_uses_two_rate_evaluations(self):
        model = FakeMaskedModel()
        output, trace = approximate.generate_trapezoidal(
            rate_function=ctmc.llada_qrate,
            steps=4,
            **common(model),
        )
        self.assertNotIn(6, output[0, 2:].tolist())
        self.assertEqual(trace["method"], "approximate_trapezoidal")
        self.assertGreater(trace["metrics"]["sequential_target_rounds"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
