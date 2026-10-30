"""Map a rollout dump's decoded input text back to its training-data row.

Several problems share their first few hundred characters, so the match uses the full user message.
"""

import pandas as pd


class PromptIndex:
    def __init__(self, data: pd.DataFrame, prefix_chars: int = 300):
        self.prefix_chars = prefix_chars
        self.by_prefix: dict[str, list[int]] = {}
        self.user_texts = [list(p)[-1]["content"] for p in data["prompt"]]
        for i, text in enumerate(self.user_texts):
            self.by_prefix.setdefault(text[:prefix_chars], []).append(i)
        self.prefixes = list(self.by_prefix)

    def row_of(self, input_text: str) -> int:
        """Index of the row whose full user message occurs in input_text."""
        matches = [
            i for prefix in self.prefixes if prefix in input_text
            for i in self.by_prefix[prefix] if self.user_texts[i] in input_text
        ]
        assert len(matches) == 1, f"expected one matching prompt, found {len(matches)}"
        return matches[0]
