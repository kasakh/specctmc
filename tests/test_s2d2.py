from __future__ import annotations

import inspect
import unittest
from collections import defaultdict

import torch

from specctmc import s2d2 as exact


class ExactCorrectionTests(unittest.TestCase):
    def assertTensorClose(self, left, right, atol=1e-11):
        self.assertTrue(torch.allclose(torch.as_tensor(left), torch.as_tensor(right), atol=atol, rtol=0.0))

    def test_temperature_one_rows_are_normalized_and_mask_is_excluded(self):
        logits = torch.tensor([[0.0, 1.0, 7.0], [2.0, -1.0, 9.0]])
        rows = exact.normalized_token_rows(logits, temperature=1.0, forbidden_token_ids=(2,))
        self.assertTensorClose(rows.sum(dim=1), torch.ones(2, dtype=torch.float64))
        self.assertTensorClose(rows[:, 2], torch.zeros(2, dtype=torch.float64))
        self.assertTrue(bool((rows[:, :2] > 0).all()))

    def test_temperature_zero_is_a_point_mass_not_softmax_confidence(self):
        logits = torch.tensor([[0.0, 1.0, -2.0], [3.0, 2.0, 1.0]])
        rows = exact.normalized_token_rows(logits, temperature=0.0)
        self.assertEqual(rows.tolist(), [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(float(rows[0, 1]), 1.0)
        self.assertLess(float(torch.softmax(logits[0].double(), dim=0)[1]), 1.0)

    def test_explicit_position_laws_are_normalized_over_active_positions(self):
        active = torch.tensor([True, False, True])
        uniform = exact.normalized_position_law(active)
        scored = exact.normalized_position_law(active, torch.tensor([1.0, 100.0, 3.0]))
        self.assertTensorClose(uniform, torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64))
        self.assertTensorClose(scored, torch.tensor([0.25, 0.0, 0.75], dtype=torch.float64))

    def test_one_forward_draft_records_every_probability(self):
        logits = torch.tensor([[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 1.0, -1.0]])
        draft = exact.draft_from_one_forward(
            logits,
            torch.tensor([True, True, True]),
            depth=3,
            uniforms=(0.05, 0.55, 0.95),
            token_temperature=1.0,
            forbidden_token_ids=(2,),
        )
        self.assertEqual(len(draft), 3)
        for record in draft:
            saved = exact.full_probability_record(record)
            self.assertEqual(len(saved["position_probs"]), 3)
            self.assertEqual(len(saved["token_probs"]), 3)
            self.assertEqual(len(saved["joint_probs"]), 3)
            self.assertEqual(saved["event_count"], 9)
            self.assertAlmostEqual(sum(sum(row) for row in saved["joint_probs"]), 1.0, places=12)
            self.assertAlmostEqual(saved["selected_probability"], record.law.mass(record.event), places=12)

    def test_finite_state_maximal_coupling_enumerates_to_target(self):
        active = torch.tensor([True])
        q = exact.event_law(
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([[0.55, 0.35, 0.10]], dtype=torch.float64),
            active,
        )
        p = exact.event_law(
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([[0.20, 0.50, 0.30]], dtype=torch.float64),
            active,
        )
        overlap = torch.minimum(q.joint_probs, p.joint_probs)
        residual = (p.joint_probs - q.joint_probs).clamp_min(0.0)
        enumerated = overlap + residual
        self.assertTensorClose(enumerated, p.joint_probs)
        self.assertAlmostEqual(float(overlap.sum() + residual.sum()), 1.0, places=12)

    def test_support_gap_is_reached_by_positive_residual(self):
        active = torch.tensor([True])
        q = exact.event_law(torch.tensor([1.0]), torch.tensor([[1.0, 0.0]]), active)
        p = exact.event_law(torch.tensor([1.0]), torch.tensor([[0.25, 0.75]]), active)
        proposal = exact.DraftEvent((0, 0), 1.0, q)
        result = exact.maximal_coupling_step(
            proposal,
            p,
            acceptance_uniform=0.9,
            residual_uniform=0.2,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.event, (0, 1))
        self.assertAlmostEqual(result.residual_mass, 0.75, places=12)

    def test_first_rejection_stops_the_current_draft(self):
        active = torch.tensor([True, True])
        logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        draft = exact.draft_from_one_forward(logits, active, 2, (0.1, 0.2))

        def target_for_prefix(prefix, current_active):
            del prefix
            rows = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            return exact.event_law(exact.normalized_position_law(current_active), rows, current_active)

        corrected = exact.correct_draft(active, draft, target_for_prefix, (0.99, 0.0), (0.1, 0.1))
        self.assertEqual(len(corrected), 1)
        self.assertFalse(corrected[0].accepted)

    def test_multistep_finite_state_enumeration_matches_direct_target(self):
        empty = -1
        initial = (empty, empty, empty)

        def logits_for_state(state):
            filled = sum(token != empty for token in state)
            base = torch.tensor(
                [
                    [2.2 + 0.1 * filled, 0.3],
                    [0.4, 1.8 + 0.2 * filled],
                    [1.1, 0.9 + 0.15 * filled],
                ],
                dtype=torch.float64,
            )
            if state[0] == 1:
                base[2] = torch.tensor([0.2, 2.4])
            return base

        def target_for_state(state):
            active = torch.tensor([token == empty for token in state])
            return exact.natural_event_target(logits_for_state(state), active)

        def apply_event(state, event):
            updated = list(state)
            self.assertEqual(updated[event[0]], empty)
            updated[event[0]] = event[1]
            return tuple(updated)

        def direct_distribution(state, steps):
            if steps == 0:
                return {state: 1.0}
            law = target_for_state(state)
            out = defaultdict(float)
            for position in range(law.joint_probs.shape[0]):
                for token in range(law.joint_probs.shape[1]):
                    mass = float(law.joint_probs[position, token])
                    if mass == 0.0:
                        continue
                    for final, tail_mass in direct_distribution(apply_event(state, (position, token)), steps - 1).items():
                        out[final] += mass * tail_mass
            return dict(out)

        def corrected_distribution(state, steps, frozen_logits=None, depth_left=0):
            if steps == 0:
                return {state: 1.0}
            active = torch.tensor([token == empty for token in state])
            if depth_left == 0:
                frozen_logits = logits_for_state(state)
                depth_left = min(2, steps)
            proposal = exact.s2d2_proposal_law(frozen_logits, active)
            target = target_for_state(state)
            out = defaultdict(float)
            overlap = torch.minimum(proposal.joint_probs, target.joint_probs)
            residual = (target.joint_probs - proposal.joint_probs).clamp_min(0.0)
            for position in range(proposal.joint_probs.shape[0]):
                for token in range(proposal.joint_probs.shape[1]):
                    event = (position, token)
                    accepted_mass = float(overlap[event])
                    corrected_mass = float(residual[event])
                    next_state = apply_event(state, event) if accepted_mass + corrected_mass else None
                    if accepted_mass:
                        for final, tail_mass in corrected_distribution(
                            next_state, steps - 1, frozen_logits, depth_left - 1
                        ).items():
                            out[final] += accepted_mass * tail_mass
                    if corrected_mass:
                        for final, tail_mass in corrected_distribution(next_state, steps - 1).items():
                            out[final] += corrected_mass * tail_mass
            return dict(out)

        direct = direct_distribution(initial, 3)
        corrected = corrected_distribution(initial, 3)
        self.assertEqual(set(direct), set(corrected))
        self.assertAlmostEqual(sum(direct.values()), 1.0, places=11)
        self.assertAlmostEqual(sum(corrected.values()), 1.0, places=11)
        for state in direct:
            self.assertAlmostEqual(direct[state], corrected[state], places=11)

    def test_no_combinatorial_sequence_marginal_is_constructed(self):
        source = inspect.getsource(exact)
        self.assertNotIn("itertools.product", source)
        self.assertNotIn("cartesian_prod", source)
        logits = torch.zeros((32, 17))
        draft = exact.draft_from_one_forward(
            logits,
            torch.ones(32, dtype=torch.bool),
            depth=4,
            uniforms=(0.1, 0.2, 0.3, 0.4),
        )
        self.assertEqual(len(draft), 4)
        self.assertTrue(all(record.law.joint_probs.shape == (32, 17) for record in draft))
        self.assertTrue(all(record.law.joint_probs.ndim == 2 for record in draft))


    def test_released_adapter_runs_end_to_end_on_cpu_fake_model(self):
        from types import SimpleNamespace
        from specctmc.s2d2_adapter import (
            generate_direct_natural_event,
            generate_exact_s2d2_proposal,
        )

        class FakeModel:
            device = torch.device("cpu")
            dtype = torch.float32

            def __init__(self):
                self.calls = 0

            def __call__(self, tokens, **kwargs):
                del kwargs
                self.calls += 1
                batch, length = tokens.shape
                logits = torch.empty((batch, length, 4), dtype=torch.float32)
                for row in range(batch):
                    state_sum = int(tokens[row].sum().item())
                    for position in range(length):
                        logits[row, position] = torch.tensor(
                            [1.7 + 0.1 * position, 1.1 + 0.05 * state_sum, 0.2, -2.0]
                        )
                return SimpleNamespace(logits=logits)

        prompt = torch.tensor([[0, 1]])
        direct_model = FakeModel()
        direct_output, direct_stats = generate_direct_natural_event(
            model=direct_model,
            input_ids=prompt,
            block_length=2,
            gen_length=2,
            eos_early_stop=False,
            mask_id=3,
            eos_id=2,
            seed=9,
        )
        self.assertEqual(direct_output.shape, (1, 4))
        self.assertNotIn(3, direct_output[0, 2:].tolist())
        self.assertEqual(direct_stats["target_calls"], 2)
        self.assertEqual(direct_stats["nfe"], direct_model.calls)

        proposal_model = FakeModel()
        proposal_output, proposal_stats = generate_exact_s2d2_proposal(
            model=proposal_model,
            input_ids=prompt,
            block_length=2,
            gen_length=2,
            eos_early_stop=False,
            mask_id=3,
            eos_id=2,
            depth=2,
            seed=9,
            record_full_proposal=True,
        )
        self.assertEqual(proposal_output.shape, (1, 4))
        self.assertNotIn(3, proposal_output[0, 2:].tolist())
        self.assertEqual(len(proposal_stats["events"]), 2)
        self.assertEqual(proposal_stats["target_calls"], 2)
        self.assertEqual(proposal_stats["nfe"], proposal_model.calls)
        for event in proposal_stats["events"]:
            full = event["full_proposal"]
            self.assertAlmostEqual(sum(sum(row) for row in full["joint_probs"]), 1.0, places=11)
            self.assertGreater(event["proposal_probability"], 0.0)
            self.assertGreaterEqual(event["acceptance_probability"], 0.0)
            self.assertLessEqual(event["acceptance_probability"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
