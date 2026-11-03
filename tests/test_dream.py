from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from specctmc import dream


class FakeDreamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls: list[int] = []

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(self, input_ids, attention_mask, position_ids):
        del attention_mask, position_ids
        batch_size, sequence_length = input_ids.shape
        self.calls.append(batch_size)
        logits = torch.full(
            (batch_size, sequence_length, dream.DREAM_VOCAB_SIZE),
            -12.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for position in range(sequence_length):
            logits[:, position, 1 + position] = 8.0
            logits[:, position, dream.DREAM_MASK_ID] = 10.0
        return SimpleNamespace(logits=logits)


class DreamAdapterTests(unittest.TestCase):
    def setUp(self):
        self.original_mask = dream.DREAM_MASK_ID
        self.original_vocabulary = dream.DREAM_VOCAB_SIZE
        dream.DREAM_MASK_ID = 6
        dream.DREAM_VOCAB_SIZE = 7

    def tearDown(self):
        dream.DREAM_MASK_ID = self.original_mask
        dream.DREAM_VOCAB_SIZE = self.original_vocabulary

    def test_shift_matches_the_official_position_rule(self):
        raw = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        expected = torch.cat((raw[:, :1], raw[:, :-1]), dim=1)
        self.assertTrue(torch.equal(dream.shift_dream_logits(raw), expected))

    def test_direct_and_speculative_wrappers_complete(self):
        common = {
            "prompt": torch.tensor([[1, 2]], dtype=torch.long),
            "batch_size": 1,
            "gen_length": 3,
            "block_length": 3,
            "mask_id": 6,
            "vocab_size": 7,
            "temperature": 1.0,
            "eps": 0.01,
            "seed": 11,
            "event_times_by_block": [[0.8, 0.4]],
            "target_evaluation_batch_size": 4,
            "return_trace": True,
        }
        direct_model = FakeDreamModel()
        direct_output, direct_trace = dream.generate_ctmc_exact(
            model=direct_model,
            stream_seeds={"event": 11, "target": 12},
            **common,
        )
        spec_model = FakeDreamModel()
        spec_output, spec_trace = dream.generate_ctmc_speculative(
            model=spec_model,
            speculative_depth=3,
            draft_kind="stale_logits",
            bootstrap_kind="uniform",
            stream_seeds={
                "event": 11,
                "draft": 13,
                "verification": 14,
                "residual": 15,
            },
            **common,
        )
        self.assertNotIn(6, direct_output[0, 2:].tolist())
        self.assertNotIn(6, spec_output[0, 2:].tolist())
        self.assertEqual(set(direct_model.calls), {4})
        self.assertEqual(set(spec_model.calls), {4})
        self.assertEqual(direct_trace["requests"][0]["event_times"], [0.8, 0.4])
        self.assertEqual(spec_trace["requests"][0]["event_times"], [0.8, 0.4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
