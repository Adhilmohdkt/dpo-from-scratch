import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from src.dpo_loss import dpo_loss, sequence_logps
from src.modeling import (base_model_identity, load_sft_policy, model_config_for_sft_manifest,
                          tokenizer_identity, validate_sft_tokenizer)
from src.train_dpo import build_reference_cache_metadata, reference_cache_matches
from src.training import restore_training_state, save_checkpoint


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path, "adapter.pt"))

    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        model.load_state_dict(torch.load(Path(path, "adapter.pt"), weights_only=True))
        return model


class FakeTokenizer:
    name_or_path = "fake/tokenizer"
    _commit_hash = "tok-commit-1"
    init_kwargs = {}
    special_tokens_map = {"eos_token": "<eos>", "pad_token": "<pad>"}
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "template-v1"

    def __init__(self):
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: "backend-vocab-v1")


class TestReferenceAndResume(unittest.TestCase):
    def test_manifest_pins_base_revision_and_tokenizer(self):
        tok = FakeTokenizer()
        identity = {"source": "org/model", "resolved_commit": "commit-a"}
        manifest = {
            "format_version": 1,
            "model_name": "org/model",
            "model_identity": identity,
            "model_revision": "commit-a",
            "tokenizer_identity": tokenizer_identity(tok),
        }
        pinned = model_config_for_sft_manifest({"name_or_path": "org/model", "revision": "main"}, manifest)
        self.assertEqual(pinned["revision"], "commit-a")
        validate_sft_tokenizer(tok, manifest)
        changed = FakeTokenizer()
        changed.chat_template = "changed-template"
        with self.assertRaises(ValueError):
            validate_sft_tokenizer(changed, manifest)
        with self.assertRaises(ValueError):
            model_config_for_sft_manifest({"name_or_path": "org/other", "revision": "main"}, manifest)

    def test_local_base_identity_uses_file_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            weight = model_dir / "model.safetensors"
            weight.write_bytes(b"weights-v1")
            first = base_model_identity({"name_or_path": str(model_dir)})
            weight.write_bytes(b"weights-v2")
            second = base_model_identity({"name_or_path": str(model_dir)})
            self.assertNotEqual(first, second)

    def test_sft_adapter_is_exact_frozen_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp, "base")
            base_dir.mkdir()
            base = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_positions=16, n_ctx=16, n_embd=8, n_layer=1, n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))
            base.save_pretrained(base_dir)
            policy = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))
            with torch.no_grad():
                for name, parameter in policy.named_parameters():
                    if "lora_B" in name:
                        parameter.fill_(0.02)
            policy.eval()
            sft_dir = Path(tmp, "sft")
            policy.save_pretrained(sft_dir)
            ids = torch.tensor([[1, 2, 3]])
            expected = policy(input_ids=ids).logits.detach()
            reference = load_sft_policy({"name_or_path": str(base_dir), "torch_dtype": "float32"}, str(sft_dir), trainable=False)
            reference.eval()
            actual = reference(input_ids=ids).logits.detach()
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
            self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))
            policy = load_sft_policy({"name_or_path": str(base_dir), "torch_dtype": "float32"}, str(sft_dir), trainable=True)
            chosen = torch.tensor([[1, 2, 3, 4]])
            rejected = torch.tensor([[1, 2, 3, 5]])
            response_mask = torch.tensor([[0, 0, 0, 1]])
            with torch.no_grad():
                ref_chosen = sequence_logps(reference(input_ids=chosen).logits, chosen, response_mask)
                ref_rejected = sequence_logps(reference(input_ids=rejected).logits, rejected, response_mask)
            policy.train()
            optimizer = torch.optim.SGD((p for p in policy.parameters() if p.requires_grad), lr=0.05)
            p_chosen = sequence_logps(policy(input_ids=chosen).logits, chosen, response_mask)
            p_rejected = sequence_logps(policy(input_ids=rejected).logits, rejected, response_mask)
            initial_margin = (p_chosen - p_rejected - ref_chosen + ref_rejected).detach()
            loss, _ = dpo_loss(p_chosen, p_rejected, ref_chosen, ref_rejected, beta=0.1)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters()))
            optimizer.step()
            policy.eval()
            with torch.no_grad():
                updated = sequence_logps(policy(input_ids=chosen).logits, chosen, response_mask) - sequence_logps(policy(input_ids=rejected).logits, rejected, response_mask) - ref_chosen + ref_rejected
            self.assertGreater(float(updated), float(initial_margin))

    def test_checkpoint_restores_optimizer_rng_and_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyAdapter()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            model.weight.grad = torch.tensor([1.0])
            optimizer.step()
            scheduler.step()
            metadata = {"stage": "sft", "model": "tiny", "max_length": 32}
            path = save_checkpoint(model, optimizer, 7, 2, tmp, {"batch_in_epoch": 5}, scheduler, metadata)
            saved_weight = model.weight.detach().clone()
            model.weight.data.add_(10)
            rng = torch.get_rng_state().clone()
            step, epoch, extra = restore_training_state(optimizer, path, scheduler, metadata)
            self.assertEqual((step, epoch, extra["batch_in_epoch"]), (7, 2, 5))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)
            resumed = TinyAdapter.from_pretrained(path)
            self.assertTrue(torch.equal(resumed.weight.detach(), saved_weight))
            with self.assertRaises(ValueError):
                restore_training_state(optimizer, path, scheduler, {"stage": "sft", "model": "changed"})

    def test_reference_cache_identity_covers_checkpoint_tokenizer_data_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            sft_dir = Path(tmp, "sft")
            sft_dir.mkdir()
            (sft_dir / "adapter_config.json").write_text("config", encoding="utf-8")
            (sft_dir / "adapter_model.safetensors").write_bytes(b"adapter-v1")

            class TrainDataset:
                rows = SimpleNamespace(_fingerprint="dataset-fingerprint-1")
                _fingerprint = "dataset-fingerprint-1"
                def __len__(self):
                    return 2

            cfg = {
                "seed": 42,
                "model": {"name_or_path": "org/model", "revision": "commit-a"},
                "data": {"name": "org/data", "revision": "data-a", "train_preferences_split": "train_prefs", "max_train_samples": None, "max_validation_samples": 20, "validation_fraction": 0.02, "max_length": 128},
            }
            tokenizer = FakeTokenizer()
            dataset = TrainDataset()
            identity = {"source": "org/model", "resolved_commit": "commit-a"}
            expected = build_reference_cache_metadata(cfg, sft_dir, dataset, tokenizer, identity)
            cache = Path(tmp, "reference.pt")
            metadata_path = Path(tmp, "reference.pt.json")
            torch.save(torch.ones((2, 2)), cache)
            metadata_path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertTrue(reference_cache_matches(cache, metadata_path, expected, 2))

            changed_tokenizer = FakeTokenizer()
            changed_tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "backend-vocab-v2")
            changed = build_reference_cache_metadata(cfg, sft_dir, dataset, changed_tokenizer, identity)
            self.assertFalse(reference_cache_matches(cache, metadata_path, changed, 2))

            changed_model = build_reference_cache_metadata(cfg, sft_dir, dataset, tokenizer, {"source": "org/model", "resolved_commit": "commit-b"})
            self.assertFalse(reference_cache_matches(cache, metadata_path, changed_model, 2))

            changed_cfg = {**cfg, "data": {**cfg["data"], "revision": "data-b", "max_length": 96}}
            changed_data = build_reference_cache_metadata(changed_cfg, sft_dir, dataset, tokenizer, identity)
            self.assertFalse(reference_cache_matches(cache, metadata_path, changed_data, 2))

            dataset.rows._fingerprint = "dataset-fingerprint-2"
            changed_fingerprint = build_reference_cache_metadata(cfg, sft_dir, dataset, tokenizer, identity)
            self.assertFalse(reference_cache_matches(cache, metadata_path, changed_fingerprint, 2))

            (sft_dir / "adapter_model.safetensors").write_bytes(b"adapter-v2")
            changed_adapter = build_reference_cache_metadata(cfg, sft_dir, dataset, tokenizer, identity)
            self.assertFalse(reference_cache_matches(cache, metadata_path, changed_adapter, 2))

            cache.write_bytes(b"corrupt")
            self.assertFalse(reference_cache_matches(cache, metadata_path, expected, 2))


if __name__ == "__main__":
    unittest.main()
