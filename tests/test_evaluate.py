import unittest

from transformers import BatchEncoding

from src.evaluate import _prompt_ids


class Transformers5Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return BatchEncoding({"input_ids": [[11, 12, 13]]})


class TestEvaluationPromptEncoding(unittest.TestCase):
    def test_prompt_ids_normalize_transformers5_batch_encoding(self):
        row = {
            "chosen": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
        ids = _prompt_ids(row, Transformers5Tokenizer())
        self.assertEqual(ids, [11, 12, 13])


if __name__ == "__main__":
    unittest.main()
