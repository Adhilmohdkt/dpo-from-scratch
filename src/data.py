"""Dataset adapters and chat-template-aware tokenization."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

PREPROCESSING_VERSION = "chat-template-last-assistant-eos-mask-v2"


def _messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "assistant", "content": value}]
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in value]


def _encode_conversation(tokenizer, messages: list[dict[str, str]], max_length: int):
    if max_length < 2:
        raise ValueError("max_length must be at least 2 to score a response token")
    assistant_positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_positions:
        raise ValueError("Conversation has no assistant response")
    # The supported UltraFeedback rows are single-turn: use the final assistant
    # answer as the target and keep any preceding turns in the prompt.
    answer_at = assistant_positions[-1]
    prompt = messages[:answer_at]
    if not prompt:
        raise ValueError("Conversation has no user/context prompt")
    prefix_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(prefix_ids, torch.Tensor):
        prefix_ids = prefix_ids.tolist()
    if isinstance(full_ids, torch.Tensor):
        full_ids = full_ids.tolist()
    # Padding/truncation are disabled above; verify the generation prefix is
    # exactly the beginning of the fully rendered conversation.
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenizer chat template does not preserve the prompt prefix")
    response_ids = full_ids[len(prefix_ids) :]
    if not response_ids:
        raise ValueError("Empty assistant response after chat-template rendering")
    # Some model templates leave EOS out; make termination scoring explicit.
    if tokenizer.eos_token_id is not None and response_ids[-1] != tokenizer.eos_token_id:
        response_ids.append(tokenizer.eos_token_id)
    if len(prefix_ids) + len(response_ids) > max_length:
        # Preserve a meaningful suffix of the prompt (including the assistant
        # generation header) when a completion alone nearly fills the budget.
        prompt_reserve = min(len(prefix_ids), max(1, min(64, max_length // 4)))
        response_budget = max_length - prompt_reserve
        if len(response_ids) > response_budget:
            response_ids = response_ids[:response_budget]
            if tokenizer.eos_token_id is not None:
                response_ids[-1] = tokenizer.eos_token_id
            prefix_ids = prefix_ids[-prompt_reserve:]
        else:
            prefix_ids = prefix_ids[-(max_length - len(response_ids)) :]
    ids = prefix_ids + response_ids
    mask = [0] * len(prefix_ids) + [1] * len(response_ids)
    return ids, mask


class PreferenceDataset(Dataset):
    """Preference examples with score metadata retained for future analysis."""

    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        chosen, chosen_mask = _encode_conversation(self.tokenizer, _messages(row["chosen"]), self.max_length)
        rejected, rejected_mask = _encode_conversation(self.tokenizer, _messages(row["rejected"]), self.max_length)
        return {
            "index": index,
            "chosen_input_ids": chosen,
            "chosen_response_mask": chosen_mask,
            "rejected_input_ids": rejected,
            "rejected_response_mask": rejected_mask,
            "prompt": row.get("prompt", ""),
            "score_chosen": row.get("score_chosen"),
            "score_rejected": row.get("score_rejected"),
        }


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        messages = _messages(row.get("messages", row.get("chosen")))
        ids, mask = _encode_conversation(self.tokenizer, messages, self.max_length)
        return {"input_ids": ids, "response_mask": mask, "index": index}


def _slice_rows(rows, count, seed: int = 42):
    # A fixed one-time shuffle gives all runs a stable but non-source-order
    # sequence, which is easy to replay exactly from a batch cursor.
    shuffled = rows.shuffle(seed=seed)
    return shuffled if count is None else shuffled.select(range(min(int(count), len(shuffled))))


def _validation_membership(row: dict, seed: int, fraction: float) -> bool:
    identity = row.get("prompt_id") or row.get("prompt")
    if identity is None:
        raise ValueError("Training rows need prompt_id or prompt to make a reproducible validation split")
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def split_train_validation(rows, fraction: float, seed: int):
    """Partition by stable prompt identity so SFT/DPO share held-out train prompts."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if fraction == 0:
        return rows, rows.select([])
    if len(rows) < 2:
        raise ValueError("At least two training prompts are required for a nonzero validation split")
    validation_indices = [i for i, row in enumerate(rows) if _validation_membership(row, seed, fraction)]
    if not validation_indices:
        # For small smoke-test subsets, deterministically reserve one row.
        validation_indices = [min(range(len(rows)), key=lambda i: hashlib.sha256(f"{seed}:{rows[i].get('prompt_id', rows[i].get('prompt', i))}".encode()).digest())]
    validation_set = set(validation_indices)
    train_indices = [i for i in range(len(rows)) if i not in validation_set]
    if not train_indices:
        raise ValueError("Validation split consumed all training examples")
    return rows.select(train_indices), rows.select(validation_indices)


def _require_columns(dataset, required: set[str], split_name: str):
    present = set(dataset.column_names)
    missing = required - present
    if missing:
        raise ValueError(f"Split {split_name!r} is missing required columns: {sorted(missing)}")


def load_preference_data(cfg: dict, tokenizer):
    name = cfg["name"]
    kwargs = {"revision": cfg["revision"]} if cfg.get("revision") else {}
    split_name = cfg.get("train_preferences_split", "train_prefs")
    rows = load_dataset(name, split=split_name, **kwargs)
    _require_columns(rows, {"chosen", "rejected", "prompt_id"}, split_name)
    seed = int(cfg.get("seed", 42))
    train, valid = split_train_validation(rows, float(cfg.get("validation_fraction", 0.02)), seed)
    train = _slice_rows(train, cfg.get("max_train_samples"), seed)
    valid = _slice_rows(valid, cfg.get("max_validation_samples"), seed + 1)
    max_len = int(cfg.get("max_length", 1024))
    return PreferenceDataset(train, tokenizer, max_len), PreferenceDataset(valid, tokenizer, max_len)


def load_preference_test_data(cfg: dict, tokenizer):
    """Load the final held-out preference split for final evaluation only."""
    name = cfg["name"]
    kwargs = {"revision": cfg["revision"]} if cfg.get("revision") else {}
    split_name = cfg.get("eval_preferences_split", "test_prefs")
    rows = load_dataset(name, split=split_name, **kwargs)
    _require_columns(rows, {"chosen", "rejected", "prompt_id"}, split_name)
    rows = _slice_rows(rows, cfg.get("max_eval_samples"), int(cfg.get("seed", 42)) + 2)
    return PreferenceDataset(rows, tokenizer, int(cfg.get("max_length", 1024)))


def load_sft_data(cfg: dict, tokenizer):
    name = cfg["name"]
    kwargs = {"revision": cfg["revision"]} if cfg.get("revision") else {}
    split_name = cfg.get("sft_split", "train_sft")
    rows = load_dataset(name, split=split_name, **kwargs)
    _require_columns(rows, {"messages", "prompt_id"}, split_name)
    seed = int(cfg.get("seed", 42))
    train, valid = split_train_validation(rows, float(cfg.get("validation_fraction", 0.02)), seed)
    train = _slice_rows(train, cfg.get("max_train_samples"), seed)
    valid = _slice_rows(valid, cfg.get("max_validation_samples"), seed + 1)
    max_len = int(cfg.get("max_length", 1024))
    return SFTDataset(train, tokenizer, max_len), SFTDataset(valid, tokenizer, max_len)


def _pad(sequences: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
    width = max(len(s) for s in sequences)
    out = torch.full((len(sequences), width), pad_value, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        out[i, : len(sequence)] = torch.as_tensor(sequence, dtype=torch.long)
    return out


def collate_preferences(batch, pad_token_id: int):
    chosen_ids = _pad([x["chosen_input_ids"] for x in batch], pad_token_id)
    rejected_ids = _pad([x["rejected_input_ids"] for x in batch], pad_token_id)
    chosen_attention = torch.zeros_like(chosen_ids, dtype=torch.bool)
    rejected_attention = torch.zeros_like(rejected_ids, dtype=torch.bool)
    for i, item in enumerate(batch):
        chosen_attention[i, : len(item["chosen_input_ids"])] = True
        rejected_attention[i, : len(item["rejected_input_ids"])] = True
    return {
        "indices": torch.tensor([x["index"] for x in batch], dtype=torch.long),
        "chosen_input_ids": chosen_ids,
        "chosen_response_mask": _pad([x["chosen_response_mask"] for x in batch], 0).bool(),
        "chosen_attention_mask": chosen_attention,
        "rejected_input_ids": rejected_ids,
        "rejected_response_mask": _pad([x["rejected_response_mask"] for x in batch], 0).bool(),
        "rejected_attention_mask": rejected_attention,
        "prompts": [x["prompt"] for x in batch],
        "score_chosen": [x["score_chosen"] for x in batch],
        "score_rejected": [x["score_rejected"] for x in batch],
    }


def collate_sft(batch, pad_token_id: int):
    ids = _pad([x["input_ids"] for x in batch], pad_token_id)
    mask = _pad([x["response_mask"] for x in batch], 0).bool()
    attention = torch.zeros_like(ids, dtype=torch.bool)
    for i, item in enumerate(batch):
        attention[i, : len(item["input_ids"])] = True
    return {"input_ids": ids, "response_mask": mask, "attention_mask": attention}
