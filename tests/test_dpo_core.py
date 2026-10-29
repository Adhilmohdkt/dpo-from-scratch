import unittest
from types import SimpleNamespace

import torch
from trl.trainer.dpo_trainer import DPOTrainer

from src.data import _encode_conversation, collate_preferences, split_train_validation
from src.dpo_loss import dpo_loss, sequence_logps


class FixedTemplateTokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        prompt = [1, 2, 3]
        answer = [4, 5, self.eos_token_id]
        return prompt if add_generation_prompt else prompt + answer


class LongTemplateTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        prefix = [1, 2, 3, 4, 5]
        return prefix if add_generation_prompt else prefix + [10, 11, 12, 13, 14, 15, 16, 17, 99]


class TestDpoCore(unittest.TestCase):
    def test_dpo_loss_matches_closed_form(self):
        pc = torch.tensor([-2.0, -1.0])
        pr = torch.tensor([-3.0, -2.0])
        rc = torch.tensor([-2.5, -1.5])
        rr = torch.tensor([-2.5, -2.5])
        beta = 0.2
        expected_logits = beta * ((pc - pr) - (rc - rr))
        loss, metrics = dpo_loss(pc, pr, rc, rr, beta)
        expected = torch.nn.functional.softplus(-expected_logits).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(metrics["logits"], expected_logits))

    def test_matches_installed_trl_sigmoid_loss(self):
        pc = torch.tensor([-2.1, -1.3, -3.0])
        pr = torch.tensor([-3.4, -2.8, -3.2])
        rc = torch.tensor([-2.0, -1.7, -3.3])
        rr = torch.tensor([-2.5, -2.9, -3.0])
        beta = 0.17
        ours, _ = dpo_loss(pc, pr, rc, rr, beta)
        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            reference_free=False,
            f_divergence_type="reverse_kl",
            f_divergence_params=None,
            beta=beta,
            loss_type="sigmoid",
            label_smoothing=0.0,
        )
        trl_losses = DPOTrainer.dpo_loss(trainer, pc, pr, rc, rr)[0]
        self.assertTrue(torch.allclose(ours, trl_losses.mean(), atol=1e-7))

    def test_identical_responses_have_half_preference_probability(self):
        x = torch.tensor([-4.2], requires_grad=True)
        loss, metrics = dpo_loss(x, x, x.detach(), x.detach(), beta=0.1)
        self.assertAlmostEqual(float(loss), torch.log(torch.tensor(2.0)).item(), places=6)
        self.assertEqual(float(metrics["reward_accuracy"].item()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_reference_logps_are_detached_from_loss(self):
        pc = torch.tensor([-1.0], requires_grad=True)
        pr = torch.tensor([-2.0], requires_grad=True)
        rc = torch.tensor([-1.5], requires_grad=True)
        rr = torch.tensor([-2.5], requires_grad=True)
        loss, _ = dpo_loss(pc, pr, rc, rr)
        loss.backward()
        self.assertIsNotNone(pc.grad)
        self.assertIsNotNone(pr.grad)
        self.assertIsNone(rc.grad)
        self.assertIsNone(rr.grad)

    def test_extreme_margins_stay_finite_and_nan_is_rejected(self):
        high = torch.tensor([1.0e20])
        low = torch.tensor([-1.0e20])
        loss, _ = dpo_loss(high, low, low, high, beta=0.5)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(FloatingPointError):
            dpo_loss(torch.tensor([float("nan")]), low, low, high, beta=0.5)

    def test_prompt_mask_and_eos(self):
        tokenizer = FixedTemplateTokenizer()
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        ids, mask = _encode_conversation(tokenizer, messages, max_length=10)
        self.assertEqual(ids, [1, 2, 3, 4, 5, 9])
        self.assertEqual(mask, [0, 0, 0, 1, 1, 1])

    def test_long_response_truncation_preserves_prompt_suffix_and_eos(self):
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        ids, mask = _encode_conversation(LongTemplateTokenizer(), messages, max_length=8)
        self.assertEqual(ids, [4, 5, 10, 11, 12, 13, 14, 99])
        self.assertEqual(mask, [0, 0, 1, 1, 1, 1, 1, 1])

    def test_prompt_id_split_is_reproducible_and_disjoint(self):
        from datasets import Dataset
        rows = Dataset.from_list([{"prompt_id": f"id-{i}", "prompt": str(i)} for i in range(100)])
        train_a, valid_a = split_train_validation(rows, 0.2, 7)
        train_b, valid_b = split_train_validation(rows, 0.2, 7)
        train_ids, valid_ids = set(train_a["prompt_id"]), set(valid_a["prompt_id"])
        self.assertTrue(train_ids.isdisjoint(valid_ids))
        self.assertEqual(valid_ids, set(valid_b["prompt_id"]))
        self.assertEqual(train_ids, set(train_b["prompt_id"]))
        reversed_rows = Dataset.from_list(list(reversed([rows[i] for i in range(len(rows))])))
        _, sft_valid = split_train_validation(reversed_rows, 0.2, 7)
        self.assertEqual(valid_ids, set(sft_valid["prompt_id"]))

    def test_collator_marks_real_tokens_and_padding_separately(self):
        batch = [
            {"index": 0, "chosen_input_ids": [1, 2, 3], "chosen_response_mask": [0, 0, 1], "rejected_input_ids": [1, 2], "rejected_response_mask": [0, 1], "prompt": "p0", "score_chosen": 8, "score_rejected": 4},
            {"index": 1, "chosen_input_ids": [1, 2], "chosen_response_mask": [0, 1], "rejected_input_ids": [1, 2, 3, 4], "rejected_response_mask": [0, 0, 1, 1], "prompt": "p1", "score_chosen": 9, "score_rejected": 3},
        ]
        result = collate_preferences(batch, pad_token_id=0)
        self.assertEqual(result["chosen_attention_mask"].tolist(), [[True, True, True], [True, True, False]])
        self.assertEqual(result["chosen_response_mask"].tolist(), [[False, False, True], [False, True, False]])
        self.assertEqual(result["rejected_attention_mask"].tolist(), [[True, True, False, False], [True, True, True, True]])

    def test_padding_tokens_are_excluded(self):
        logits = torch.zeros((1, 5, 10))
        ids = torch.tensor([[1, 2, 3, 0, 0]])
        response_mask = torch.tensor([[0, 0, 1, 0, 0]])
        score = sequence_logps(logits, ids, response_mask)
        expected = torch.log(torch.tensor(0.1))
        self.assertTrue(torch.allclose(score, expected.reshape(1)))

    def test_prompt_tokens_do_not_contribute(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        logits = torch.zeros((1, 4, 6))
        logits[0, 0, 2] = 50.0  # A prompt-token prediction must be ignored.
        mask = torch.tensor([[0, 0, 0, 1]])
        score = sequence_logps(logits, ids, mask)
        self.assertAlmostEqual(float(score.item()), -torch.log(torch.tensor(6.0)).item(), places=5)


if __name__ == "__main__":
    unittest.main()
