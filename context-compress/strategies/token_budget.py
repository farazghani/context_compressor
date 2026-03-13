"""
Token-budget strategy — the most important strategy for real API use.

Guarantees the output fits *max_tokens* by repeatedly dropping the
lowest-scored sentence until the budget is satisfied.

Use this when you MUST NOT exceed a context window.
"""

from typing import List, Optional, Tuple

from .base import BaseStrategy
from .extractive import ExtrativeStrategy
from ..tokenizers.base import BaseTokenizer


class TokenBudgetStrategy(BaseStrategy):
    name = "token_budget"

    def __init__(self, tokenizer: Optional[BaseTokenizer] = None):
        super().__init__(tokenizer)
        # Reuse extractive scorer — no point duplicating TF-IDF
        self._extractor = ExtrativeStrategy(tokenizer=self.tokenizer)

    def _compress_impl(
        self,
        text: str,
        target_ratio: float,
        token_budget: Optional[int],
    ) -> str:
        sentences = self._extractor._split_sentences(text)

        if not sentences:
            return text

        scores = self._extractor._score_sentences(sentences)

        # If no hard budget, derive from ratio
        if token_budget is None:
            total = sum(self.tokenizer.count(s) for s in sentences)
            token_budget = max(1, int(total * target_ratio))

        return self._trim_to_budget(sentences, scores, token_budget)

    def _trim_to_budget(
        self,
        sentences: List[str],
        scores: List[float],
        budget: int,
    ) -> str:
        # Work with (index, sentence, score, token_count)
        items: List[Tuple[int, str, float, int]] = [
            (i, s, scores[i], self.tokenizer.count(s))
            for i, s in enumerate(sentences)
        ]

        total_tokens = sum(item[3] for item in items)

        if total_tokens <= budget:
            return " ".join(s for _, s, _, _ in items)

        # Drop lowest-scoring sentences one at a time until we fit
        # Use a list we can pop from; sort by score ascending for easy removal
        remaining = list(items)
        remaining.sort(key=lambda x: x[2])  # ascending score

        while total_tokens > budget and len(remaining) > 1:
            dropped = remaining.pop(0)  # drop lowest score
            total_tokens -= dropped[3]

        # Restore original order
        remaining.sort(key=lambda x: x[0])
        result = " ".join(s for _, s, _, _ in remaining)

        # Final safety: hard truncate by tokenizer if still over
        if self.tokenizer.count(result) > budget:
            result = self.tokenizer.truncate_to_budget(result, budget)

        return result